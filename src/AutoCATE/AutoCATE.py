import numpy as np
import optuna
import time

import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from src.AutoCATE.preprocessors import get_preprocess_pipeline
from src.AutoCATE.baselearners import get_base_learner
from src.AutoCATE.evaluators import (DREvaluator, REvaluator, NNEvaluator, ZEvaluator, UEvaluator, FEvaluator,
                                     TEvaluator, IFEvaluator)
from src.AutoCATE.metalearners import metalearner_collector
from src.AutoCATE.utils import (ContStratifiedKFold, ContStratifiedKFoldWithTreatment, ConstrainedRegressor,
                                SingleStratifiedSplitWithTreatment)


optuna.logging.set_verbosity(optuna.logging.WARNING)


class AutoCATE():
    def __init__(self, evaluation_metrics=None, preprocessors=None, base_learners=None, metalearners=None,
                 task="regression", metric="MSE", ensemble_strategy="top1average", single_base_learner=False,
                 joint_optimization=False, n_folds=1, n_trials=50, n_eval_versions=1, n_eval_trials=50, seed=42,
                 visualize=False, max_time=None, n_jobs=-1, cross_val_predict_folds=1, holdout_ratio=0.3):

        print('Jobs:', n_jobs, '(only used for sklearn; not optuna).')
        print('Max time per study:', max_time, 'seconds.')

        # Check input:
        assert task in ["regression", "classification"], AssertionError(
            "Task must be either 'regression' or 'classification'")
        assert metric in ["R2", "D2", "MSE", "RMSE", "MAE", "MAPE", "AUQC"], AssertionError(
            "Metric must be either 'R2', 'D2', 'MSE', 'RMSE', 'MAE', 'MAPE', or 'AUQC'")

        if preprocessors is None:
            preprocessors = {
                "feature_selector": ["EmptyStep", "VarianceThreshold", "SelectPercentile"],
                # "feature_selector": ["EmptyStep", "VarianceThreshold", "SelectPercentile", "PCA", "KernelPCA",
                #                      "FastICA"]
                "feature_transformer": ["EmptyStep"],
                # "feature_transformer": ["EmptyStep", "SplineTransformer", "PolynomialFeatures", "KBinsDiscretizer"]
                "feature_scaler": ["EmptyStep", "StandardScaler", "RobustScaler"],
                # "feature_scaler": ["EmptyStep", "StandardScaler", "MinMaxScaler", "RobustScaler", "PowerTransformer",
                #                    "QuantileTransformer"]
            }
        if base_learners is None:
            base_learners = ["GB"]
            # base_learners = ["DT", "LR"]  # Fast learners
            # base_learners = ["RF", "GB", "ET"]  # Complex, relatively fast models
            # base_learners = ["RF", "GB", "ET", "NN"]  # Complex models (Best)
            # base_learners = ["RF", "GB", "ET", "NN", "LR", "DT"]  # Good options
            # base_learners = ["RF", "LR", "GB", "ET", "NN", "GP", "SVM", "kNN", "DT"]  # All models
        if evaluation_metrics is None:
            # evaluation_metrics = ["kNN"]
            evaluation_metrics = ["T"]
            # evaluation_metrics = ["kNN", "R", "DR", "Z", "U", "F", "T", "IF"]  # All options
        if metalearners is None:
            metalearners = ["T"]
            # metalearners = ["S", "T", "Lo"]  # Fast options
            # metalearners = ["S", "T", "X", "DR", "Lo", "RA"]  # Selected options (Best)
            # metalearners = ["S", "T", "X", "DR", "R", "Lo", "U", "F", "Z", "RA"]  # All options

        self.preprocessors = preprocessors
        self.evaluation_metrics = evaluation_metrics
        self.evaluators = {}
        self.base_learners = base_learners
        self.metalearners = metalearners
        self.metric = metric
        # Set optimization direction
        self.opt_direction = "maximize" if metric in ["AUQC"] else "minimize"
        assert ensemble_strategy in ["pareto", "top1average", "top5average", "top1distance", "top5distance",
                                     "top1ranking", "top5ranking", "stacking"], AssertionError(
            "Ensemble strategy must be one of 'pareto', 'top1average', 'top5average', 'top1distance',  'top5distance', "
            "'top1ranking', 'top5ranking', or 'stacking'.")
        self.ensemble_strategy = ensemble_strategy
        self.joint_optimization = joint_optimization
        self.single_base_learner = single_base_learner
        self.n_folds = n_folds

        self.task = task
        self.n_trials = n_trials
        assert n_eval_versions <= n_eval_trials, AssertionError("n_eval_versions must be smaller than n_eval_trials")
        self.n_eval_versions = n_eval_versions
        self.n_eval_trials = n_eval_trials
        self.seed = seed
        self.visualize = visualize
        self.max_time = max_time
        self.n_jobs = n_jobs

        self.cross_val_predict_folds = cross_val_predict_folds

        self.holdout_ratio = holdout_ratio  # Only used if n_folds == 1

        self.z_score_threshold = np.inf
        self.CLIP = 1e-6

        self.cv_splitter = None

    def fit(self, X, t, y):
        """
        Function implementing the AutoCATE training algorithm:
        1) Construct evaluators
        2) Search the best estimators
        3) Build the final ensemble
        :param X: features
        :param t: treatment
        :param y: outcome
        """

        # Create folds and associated cv splitter, stratified on y and t
        if self.n_folds > 1:
            skf = ContStratifiedKFoldWithTreatment(n_splits=self.n_folds)
        else:
            skf = SingleStratifiedSplitWithTreatment(random_state=self.seed, test_size=self.holdout_ratio)
        train_folds = []
        val_folds = []
        for i, (train_index, val_index) in enumerate(skf.split(X, t, y)):
            train_folds.append(train_index)
            val_folds.append(val_index)

        def cv_splitter():
            for i in range(self.n_folds):
                yield train_folds[i], val_folds[i]

        self.cv_splitter = cv_splitter

        # Get evaluators
        print('\nBuilding evaluators...')
        self._get_evaluators(X, t, y)

        # Get estimators
        print('\nBuilding estimators...')
        self._get_estimators(X, t, y)

        # Build final ensemble
        print('\nBuilding ensemble...')
        self._build_ensemble(X, t, y)

    def predict(self, X, agg="mean"):
        print('\nPredicting...')
        taus = []
        for prop_model, preprocess_pipeline, model in zip(
                self.best_prop_models, self.best_preprocess_pipelines, self.best_models):
            X_trans = preprocess_pipeline.transform(X)
            # For X-Learner, propensity score is required
            if model.__class__.__name__ in ["BaseXRegressor", "BaseXClassifier"]:
                prop_est = prop_model.predict_proba(X_trans)[:, 1]
                prop_est = np.clip(prop_est, self.CLIP, 1 - self.CLIP)
                cate_pred = model.predict(X_trans, p=prop_est)
            else:
                cate_pred = model.predict(X_trans)
            taus.append(cate_pred)

        # Check if all predictions are constant
        # if np.unique(taus, return_counts=True)[1].shape[0] == 1:
        #     print('\tConstant tau predictions detected. Returning the same value for all samples.')

        # Add weighting if necessary
        if self.ensemble_strategy == "stacking":
            taus = np.array(taus)
            taus = np.average(taus, axis=0, weights=self.weights)

            return taus[:, 0]
        else:
            if agg == "mean":
                return np.mean(taus, axis=0)[:, 0]
            elif agg == "median":
                return np.median(taus, axis=0)[:, 0]

    def _get_evaluators(self, X, t, y):
        @ignore_warnings(category=ConvergenceWarning)
        def prop_objective(trial, X_eval, t_eval):
            # Define the search space for the propensity model
            base_learner_prop = trial.suggest_categorical("base_learner_prop", self.base_learners)
            prop_model = get_base_learner(trial, task="classification", base_learner=base_learner_prop,
                                          meta_learner="prop_metric", n_jobs=self.n_jobs)

            prop_model = CalibratedClassifierCV(prop_model, cv=self.n_folds if self.n_folds > 1 else 2,
                                                method=trial.suggest_categorical("method", ["sigmoid", "isotonic"]))

            # Choose preprocess pipeline and model
            prop_pipeline = get_preprocess_pipeline(trial, n_dim=X_eval.shape[1], preprocessors=self.preprocessors,
                                                    task="classification")
            prop_pipeline.steps.append(("classifier", prop_model))

            try:
                if self.cross_val_predict_folds == 1:
                    X_train, X_val, t_train, t_val = train_test_split(X_eval, t_eval,
                                                                      test_size=self.holdout_ratio,
                                                                      random_state=self.seed)
                    prop_pipeline = prop_pipeline.fit(X_train, t_train)
                    prop_est_val = prop_pipeline.predict_proba(X_val)[:, 1]
                    prop_est = prop_pipeline.predict_proba(X_eval)[:, 1]

                    trial.set_user_attr(key="prop_est", value=prop_est)

                    return np.mean((prop_est_val - t_val) ** 2)
                else:
                    skf = StratifiedKFold(n_splits=self.cross_val_predict_folds)
                    prop_est = cross_val_predict(prop_pipeline, X_eval, t_eval, cv=skf, method="predict_proba")[:, 1]

                    trial.set_user_attr(key="prop_est", value=prop_est)

                    return np.mean((prop_est - t_eval) ** 2)
            except Exception as e:
                print('Error:', e)
                return

        @ignore_warnings(category=ConvergenceWarning)
        def outcome_objective(trial, X_group, y_group, get_pipeline=False):
            outcome_pipeline = get_preprocess_pipeline(trial, n_dim=X_group.shape[1], preprocessors=self.preprocessors,
                                                       task=self.task)

            base_learner_outcome = trial.suggest_categorical("base_learner_outcome", self.base_learners)
            outcome_model = get_base_learner(trial, task=self.task, base_learner=base_learner_outcome,
                                             meta_learner="outcome_model", n_jobs=self.n_jobs)
            outcome_pipeline.steps.append(("model", outcome_model))

            if get_pipeline:
                try:
                    if self.cross_val_predict_folds == 1:
                        X_train, X_val, y_train, y_val = train_test_split(X_group, y_group,
                                                                          test_size=self.holdout_ratio,
                                                                          random_state=self.seed)
                        outcome_pipeline = outcome_pipeline.fit(X_train, y_train)
                        trial.set_user_attr(key="outcome_pipeline", value=outcome_pipeline)

                        y_pred_val = outcome_pipeline.predict(X_val)
                        return np.mean((y_pred_val - y_val) ** 2)
                    else:
                        skf = ContStratifiedKFold(n_splits=self.cross_val_predict_folds)
                        y_pred = cross_val_predict(outcome_pipeline, X_group, y_group, cv=skf, method="predict")

                        # Final pipeline: refit on entire data set
                        outcome_pipeline.fit(X_group, y_group)
                        trial.set_user_attr(key="outcome_pipeline", value=outcome_pipeline)

                        return np.mean((y_pred - y_group) ** 2)
                except Exception as e:
                    print(f"\tTrial failed due to {e}")
                    return
            else:
                try:
                    if self.cross_val_predict_folds == 1:
                        X_train, X_val, y_train, y_val = train_test_split(X_group, y_group,
                                                                          test_size=self.holdout_ratio,
                                                                          random_state=self.seed)
                        outcome_pipeline = outcome_pipeline.fit(X_train, y_train)

                        outcome_est_val = outcome_pipeline.predict(X_val)
                        outcome_est = outcome_pipeline.predict(X_group)

                        trial.set_user_attr(key="outcome_est", value=outcome_est)

                        return np.mean((outcome_est_val - y_val) ** 2)
                    else:
                        skf = ContStratifiedKFold(n_splits=self.cross_val_predict_folds)
                        outcome_est = cross_val_predict(outcome_pipeline, X_group, y_group, cv=skf, method="predict")

                        trial.set_user_attr(key="outcome_est", value=outcome_est)

                        return np.mean((outcome_est - y_group) ** 2)
                except Exception as e:
                    print(f"\tTrial failed due to {e}")
                    return

        # Obtain propensity scores if needed
        if {"R", "DR", "Z", "U", "F", "IF"}.intersection(self.evaluation_metrics):
            print('\n\tObtaining propensity scores')
            # On each validation fold, fit propensity score model and get out-of-fold estimates for the pseudo outcomes
            prop_estimates = []
            for i, (_, val_index) in enumerate(self.cv_splitter()):
                # Get validation set
                X_val, t_val = X[val_index], t[val_index]

                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=self.seed + i),
                    # sampler=optuna.samplers.TPESampler(seed=self.seed + i),
                )
                study.optimize(
                    # prop_objective,
                    lambda trial: prop_objective(trial, X_val, t_val),
                    n_trials=self.n_eval_trials, show_progress_bar=True, n_jobs=1, timeout=self.max_time
                )

                # if self.visualize:
                #     optuna.visualization.plot_optimization_history(study).show()
                #     optuna.visualization.plot_slice(study).show()

                if self.n_eval_versions == 1:
                    prop_est = [study.best_trial.user_attrs["prop_est"]]
                else:
                    trial_values = [trial.value if trial.value is not None else np.inf for trial in study.trials]
                    top_trials = np.argsort(trial_values)[:self.n_eval_versions]
                    prop_est = []
                    for trial in top_trials:
                        prop_est.append(study.trials[trial].user_attrs["prop_est"])

                prop_estimates.append(prop_est)

            # Visualize estimated propensity scores, if needed
            if self.visualize:
                for i in range(self.n_eval_versions):
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        # Get validation set
                        t_val = t[val_index]
                        plt.plot(prop_estimates[j][i], t_val, linestyle="None", marker='o')
                plt.title('Propensity score estimates')
                plt.show()

            if "Z" in self.evaluation_metrics:
                # Z-Risk:
                for i in range(self.n_eval_versions):
                    self.evaluators["Z_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["Z_" + str(i)] += [
                            ZEvaluator(X[val_index], t[val_index], y[val_index], t_pred=prop_estimates[j][i],
                                       metric=self.metric)]

            if "F" in self.evaluation_metrics:
                # F-Risk:
                for i in range(self.n_eval_versions):
                    self.evaluators["F_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["F_" + str(i)] += [
                            FEvaluator(X[val_index], t[val_index], y[val_index], t_pred=prop_estimates[j][i],
                                       metric=self.metric)]

        if {"DR", "T", "IF"}.intersection(self.evaluation_metrics):
            print('\n\tObtaining treatment-specific outcome estimates')
            outcome0_estimates = []
            outcome1_estimates = []
            for i, (_, val_index) in enumerate(self.cv_splitter()):
                # Get validation set
                X_val, t_val, y_val = X[val_index], t[val_index], y[val_index]

                # Fit the outcome models:
                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=self.seed + i),
                    # sampler=optuna.samplers.TPESampler(seed=self.seed + i),
                )
                study.optimize(
                    lambda trial: outcome_objective(trial, X_val[t_val == 0], y_val[t_val == 0], get_pipeline=True),
                    n_trials=self.n_eval_trials, show_progress_bar=True, n_jobs=1, timeout=self.max_time)

                # if self.visualize:
                #     optuna.visualization.plot_optimization_history(study).show()
                #     optuna.visualization.plot_slice(study).show()

                # Get the top n_eval_versions models
                if self.n_eval_versions == 1:
                    outcome0_model_list = [study.best_trial.user_attrs["outcome_pipeline"]]
                else:
                    trial_values = [trial.value for trial in study.trials]
                    top_trials = np.argsort(trial_values)[:self.n_eval_versions]
                    outcome0_model_list = []
                    for trial in top_trials:
                        outcome0_model_list.append(study.trials[trial].user_attrs["outcome_pipeline"])

                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=self.seed + i),
                    # sampler=optuna.samplers.TPESampler(seed=self.seed + i),
                )
                study.optimize(lambda trial: outcome_objective(trial, X_val[t_val == 1], y_val[t_val == 1], get_pipeline=True),
                               n_trials=self.n_eval_trials, show_progress_bar=True, n_jobs=1,
                               timeout=self.max_time)

                # if self.visualize:
                #     optuna.visualization.plot_optimization_history(study).show()
                #     optuna.visualization.plot_slice(study).show()

                if self.n_eval_versions == 1:
                    outcome1_model_list = [study.best_trial.user_attrs["outcome_pipeline"]]
                else:
                    trial_values = [trial.value for trial in study.trials]
                    top_trials = np.argsort(trial_values)[:self.n_eval_versions]
                    outcome1_model_list = []
                    for trial in top_trials:
                        outcome1_model_list.append(study.trials[trial].user_attrs["outcome_pipeline"])

                # Predict for all instances for the top n_eval_versions models
                outcome0_estimates_fold = []
                outcome1_estimates_fold = []
                # for i in range(self.n_eval_ensemble):
                for i in range(self.n_eval_versions):
                    outcome0_model = outcome0_model_list[i]
                    outcome0_estimates_fold.append(outcome0_model.predict(X_val))

                    outcome1_model = outcome1_model_list[i]
                    outcome1_estimates_fold.append(outcome1_model.predict(X_val))

            #     outcome0_estimates_fold.append(np.mean(outcome0_ensemble))
            #     outcome1_estimates_fold.append(np.mean(outcome1_ensemble))
            # outcome0_estimates.append(outcome0_estimates_fold)
            # outcome1_estimates.append(outcome1_estimates_fold)
                outcome0_estimates.append(outcome0_estimates_fold)
                outcome1_estimates.append(outcome1_estimates_fold)

            # Visualize estimated propensity scores, if needed
            if self.visualize:
                for i in range(self.n_eval_versions):
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        # Get validation set
                        t_val, y_val = t[val_index], y[val_index]
                        plt.plot(outcome0_estimates[j][i][t_val == 0], y_val[t_val == 0], linestyle="None", marker='o',
                                 color="red")
                        plt.plot(outcome1_estimates[j][i][t_val == 1], y_val[t_val == 1], linestyle="None", marker='o',
                                 color="blue")
                plt.title('Group-specific outcome estimates')
                plt.plot(np.unique(y_val), np.unique(y_val), 'b', linestyle='--', alpha=0.2)
                plt.show()

            if "DR" in self.evaluation_metrics:
                for i in range(self.n_eval_versions):
                    self.evaluators["DR_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["DR_" + str(i)] += [
                            DREvaluator(X[val_index], t[val_index], y[val_index], t_pred=prop_estimates[j][i],
                                        y0_pred=outcome0_estimates[j][i], y1_pred=outcome1_estimates[j][i],
                                        metric=self.metric)]

            if "T" in self.evaluation_metrics:
                for i in range(self.n_eval_versions):
                    self.evaluators["T_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["T_" + str(i)] += [
                            TEvaluator(X[val_index], t[val_index], y[val_index], y0_pred=outcome0_estimates[j][i],
                                       y1_pred=outcome1_estimates[j][i], metric=self.metric)]

            if "IF" in self.evaluation_metrics:
                for i in range(self.n_eval_versions):
                    self.evaluators["IF_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["IF_" + str(i)] += [
                            IFEvaluator(X[val_index], t[val_index], y[val_index], t_pred=prop_estimates[j][i],
                                        y0_pred=outcome0_estimates[j][i], y1_pred=outcome1_estimates[j][i])]

        if "R" in self.evaluation_metrics or "U" in self.evaluation_metrics:
            print('\n\tObtaining mu outcome estimates')
            outcome_estimates = []
            for i, (_, val_index) in enumerate(self.cv_splitter()):
                X_val, y_val = X[val_index], y[val_index]

                # Fit the outcome model
                study = optuna.create_study(
                    sampler=optuna.samplers.RandomSampler(seed=self.seed + i),
                    # sampler=optuna.samplers.TPESampler(seed=self.seed + i),
                )
                study.optimize(
                    lambda trial: outcome_objective(trial, X_group=X_val, y_group=y_val, get_pipeline=False),
                    n_trials=self.n_eval_trials, show_progress_bar=True, n_jobs=1, timeout=self.max_time)

                # if self.visualize:
                #     optuna.visualization.plot_optimization_history(study).show()
                #     optuna.visualization.plot_slice(study).show()

                if self.n_eval_versions == 1:
                    outcome_est = [study.best_trial.user_attrs["outcome_est"]]
                else:
                    trial_values = [trial.value for trial in study.trials]
                    top_trials = np.argsort(trial_values)[:self.n_eval_versions]
                    outcome_est = []
                    for trial in top_trials:
                        outcome_est.append(study.trials[trial].user_attrs["outcome_est"])

                outcome_estimates.append(outcome_est)

            if self.visualize:
                for i in range(self.n_eval_versions):
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        # Get validation set
                        y_val = y[val_index]
                        plt.plot(outcome_estimates[j][i], y_val, linestyle="None", marker='o', color='black')
                plt.title('Mu estimates')
                plt.plot(np.unique(y_val), np.unique(y_val), 'b', linestyle='--', alpha=0.2)
                plt.show()

            # R-Risk:
            if "R" in self.evaluation_metrics:
                for i in range(self.n_eval_versions):
                    self.evaluators["R_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["R_" + str(i)] += [
                            REvaluator(X[val_index], t[val_index], y[val_index],
                                       t_pred=prop_estimates[j][i],
                                       mu_pred=outcome_estimates[j][i],
                                       metric=self.metric)]

            # U-Risk
            if "U" in self.evaluation_metrics:
                for i in range(self.n_eval_versions):
                    self.evaluators["U_" + str(i)] = []
                    for j, (_, val_index) in enumerate(self.cv_splitter()):
                        self.evaluators["U_" + str(i)] += [
                            UEvaluator(X[val_index], t[val_index], y[val_index],
                                       t_pred=prop_estimates[j][i],
                                       mu_pred=outcome_estimates[j][i],
                                       metric=self.metric)]

        # NN-Risk:
        if "kNN" in self.evaluation_metrics:
            for i in range(self.n_eval_versions):
                self.evaluators[str(i + 1) + "NN"] = []
                for j, (_, val_index) in enumerate(self.cv_splitter()):
                    self.evaluators[str(i + 1) + "NN"] += [
                        NNEvaluator(X[val_index], t[val_index], y[val_index], k=i + 1, metric=self.metric)]

        # Pre-compute ATE score(s) for baseline comparison
        self.ate_scores = {}
        for evaluator in self.evaluators.keys():
            self.ate_scores[evaluator] = []
            for i, (train_index, val_index) in enumerate(self.cv_splitter()):
                # For each fold, get ate for the training set
                t_train, t_val = t[train_index], t[val_index]
                y_train, y_val = y[train_index], y[val_index]
                ate_train = np.mean(y_train[t_train == 1]) - np.mean(y_train[t_train == 0])
                ate_train = np.repeat(a=ate_train, repeats=y_val.size)

                # Score the ATE
                self.ate_scores[evaluator] += [self.evaluators[evaluator][i].score(cate_pred=ate_train)]

        if self.visualize:
            plt.figure()
            plt.title('Pseudo-outcomes')
            legend_labels = []
            pseudo_outcomes = []
            for evaluator in self.evaluators.keys():
                legend_labels.append(evaluator)
                pseudo_outcomes_evaluator = []
                for i in range(self.n_folds):
                    pseudo_outcomes_evaluator += list(self.evaluators[evaluator][i].pseudo_outcomes)
                pseudo_outcomes.append(pseudo_outcomes_evaluator)
            plt.boxplot(pseudo_outcomes)
            # Add legend labels as x-ticks
            plt.xticks(ticks=np.arange(1, len(legend_labels) + 1), labels=legend_labels)
            plt.show()

        # Drop outliers from the pseudo-outcomes, based on the z-score
        for evaluator in self.evaluators.keys():
            for i in range(self.n_folds):
                z_scores = zscore(self.evaluators[evaluator][i].pseudo_outcomes)
                self.evaluators[evaluator][i].pseudo_outcomes[z_scores > self.z_score_threshold] = np.nan
                self.evaluators[evaluator][i].pseudo_outcomes[z_scores < -self.z_score_threshold] = np.nan
                print('\tDropped', np.sum(np.isnan(self.evaluators[evaluator][i].pseudo_outcomes)), 'outliers for',
                      evaluator)

    def _get_estimators(self, X, t, y):
        # Search over different options for the CATE estimation pipeline
        @ignore_warnings(category=ConvergenceWarning)
        def objective(trial):
            # Choose preprocessing pipeline and output scaler
            preprocess_pipeline = get_preprocess_pipeline(trial, n_dim=X.shape[1], preprocessors=self.preprocessors,
                                                          task=self.task)

            # Choose metalearner
            metalearner = trial.suggest_categorical("metalearner", self.metalearners)

            # Get base learner(s), depending on metalearner
            get_metalearner = metalearner_collector(metalearner)
            cate_model = get_metalearner(trial=trial, task=self.task, base_learners=self.base_learners,
                                         joint_optimization=self.joint_optimization,
                                         single_base_learner=self.single_base_learner, n_jobs=self.n_jobs)

            eval_results = {evaluator: [] for evaluator in self.evaluators.keys()}

            for i, (train_index, val_index) in enumerate(self.cv_splitter()):
                X_train, X_val = X[train_index], X[val_index]
                t_train, t_val = t[train_index], t[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # For some metalearners, we first need to estimate the propensity scores
                if metalearner in ["R", "DR", "X", "Z", "U", "F"]:
                    base_learner_prop = trial.suggest_categorical(str(metalearner).lower() + "_base_learner_prop",
                                                                  self.base_learners)
                    prop_model = get_base_learner(trial, task="classification", base_learner=base_learner_prop,
                                                  meta_learner=str(metalearner).lower() + "_prop",
                                                  joint_optimization=self.joint_optimization, n_jobs=self.n_jobs)
                    prop_model = CalibratedClassifierCV(prop_model, cv=self.n_folds if self.n_folds > 1 else 2,
                                                        method=trial.suggest_categorical("method",
                                                                                         ["sigmoid", "isotonic"]))

                    X_train_trans = preprocess_pipeline.fit_transform(X_train, y_train)
                    if self.cross_val_predict_folds == 1:
                        prop_model.fit(X_train_trans, t_train)
                        prop_est_train = prop_model.predict_proba(X_train_trans)[:, 1]
                    else:
                        skf = StratifiedKFold(n_splits=self.cross_val_predict_folds)
                        prop_est_train = cross_val_predict(prop_model, X_train_trans, t_train, cv=skf,
                                                           method="predict_proba")[:, 1]
                    prop_est_train = np.clip(prop_est_train, self.CLIP, 1 - self.CLIP)

                try:
                    # Fit the base learner
                    X_train_trans = preprocess_pipeline.fit_transform(X_train, y_train)
                    if metalearner in ["R", "DR", "X", "Z", "U", "F"]:
                        cate_model.fit(X_train_trans, t_train, y_train, p=prop_est_train)
                    else:
                        cate_model.fit(X_train_trans, t_train, y_train)

                    # Predict the CATE and evaluate
                    X_val_trans = preprocess_pipeline.transform(X_val)
                    if metalearner == "X":
                        prop_model.fit(X_train_trans, t_train)
                        prop_est_val = prop_model.predict_proba(X_val_trans)[:, 1]
                        prop_est_val = np.clip(prop_est_val, self.CLIP, 1 - self.CLIP)
                        cate_pred_val = cate_model.predict(X_val_trans, p=prop_est_val)
                    else:
                        cate_pred_val = cate_model.predict(X_val_trans)
                except Exception as e:
                    print(f"\tTrial failed due to {e}")
                    return

                for evaluator in self.evaluators:
                    score = self.evaluators[evaluator][i].score(cate_pred=cate_pred_val)
                    # Normalize by ATE if needed
                    if self.metric in ["R2", "D2"]:
                        eval_results[evaluator].append((score - 1) / (self.ate_scores[evaluator][i] - 1))
                    elif self.metric == "AUQC":
                        eval_results[evaluator].append(score)
                    else:
                        eval_results[evaluator].append(score / self.ate_scores[evaluator][i])

            # Average evaluation results over folds:
            eval_results = [np.mean(eval_results[evaluator]) for evaluator in self.evaluators]

            # Save pipeline attributes
            trial.set_user_attr(key="metalearner", value=metalearner)
            if metalearner in ["R", "DR", "X", "Z", "U", "F"]:
                trial.set_user_attr(key="prop_model", value=prop_model)
            trial.set_user_attr(key="preprocess_pipeline", value=preprocess_pipeline)
            trial.set_user_attr(key="cate_model", value=cate_model)

            return eval_results

        study = optuna.create_study(
            directions=len(self.evaluation_metrics) * self.n_eval_versions * [self.opt_direction],
            sampler=optuna.samplers.RandomSampler(seed=self.seed),
            # sampler=optuna.samplers.TPESampler(seed=self.seed, multivariate=True, group=True),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True, n_jobs=1,
                       timeout=self.max_time)

        print('\tMetric correlations: \n', list(self.evaluators.keys()), '\n',
              np.round(np.corrcoef(
                  np.array(
                      [trial.values for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]).T
              ), 4))

        # Visualize the optimization results:
        if self.visualize:
            # for eval_iter, eval_metric in enumerate(self.evaluators.keys()):
            #     optuna.visualization.plot_optimization_history(study,
            #                                                    target=lambda t: t.values[eval_iter],
            #                                                    target_name=eval_metric).show()

            fig = optuna.visualization.plot_slice(
                study,
                target=lambda t: t.values[-1],
                target_name=list(self.evaluators.keys())[-1]
            )
            fig.update_yaxes(range=[0, 2])
            fig.show()

        # Save study
        self.study = study

    def _build_ensemble(self, X, t, y):
        start = time.time()
        if self.ensemble_strategy == "pareto":
            print("\t", len(self.study.best_trials), "best trials found.")
            # Get the best preprocessor-model combinations, i.e., all those from the best trials
            self.best_preprocess_pipelines, self.best_prop_models, self.best_models = [], [], []

            for trial in self.study.best_trials:
                preprocess_pipeline = trial.user_attrs["preprocess_pipeline"]
                prop_model = trial.user_attrs.get("prop_model", None)
                cate_model = trial.user_attrs["cate_model"]

                preprocess_pipeline.fit(X, y)
                X_trans = preprocess_pipeline.transform(X)

                if prop_model is not None:
                    prop_model.fit(X_trans, t)
                    prop_est = prop_model.predict_proba(X_trans)[:, 1]
                    prop_est = np.clip(prop_est, self.CLIP, 1 - self.CLIP)
                    cate_model.fit(X_trans, t, y, p=prop_est)
                else:
                    cate_model.fit(X_trans, t, y)

                self.best_preprocess_pipelines.append(preprocess_pipeline)
                self.best_prop_models.append(prop_model)
                self.best_models.append(cate_model)

        elif self.ensemble_strategy in ["top1average", "top1distance", "top1ranking"]:
            # Get the single best model, based on the average of the evaluation metrics:
            trial_values = np.array(
                [trial.values if trial.state == optuna.trial.TrialState.COMPLETE else [np.nan] * len(self.evaluators)
                 for trial in self.study.trials])

            if self.visualize:
                plt.boxplot(np.clip(trial_values, -1e2, 1e5), labels=self.evaluators.keys(), showfliers=False)
                plt.show()

            # Depending on strategy, calculate euclidian distance of each point to the origin or average the metrics
            if self.ensemble_strategy == "top1distance":
                avg_metrics = np.sqrt(np.sum(trial_values ** 2, axis=1))  # Euclidean norm
            elif self.ensemble_strategy == "top1average":
                avg_metrics = np.average(a=trial_values, axis=1)  # (Arithmetic) mean
            else:  # top1ranking
                rankings = np.argsort(trial_values, axis=0)
                avg_metrics = np.average(a=rankings, axis=1)

            argselector = np.nanargmax if self.opt_direction == "maximize" else np.nanargmin
            best_trial = self.study.trials[argselector(avg_metrics)]

            # Load best model and refit it on the entire training data
            best_metalearner = best_trial.user_attrs["metalearner"]
            best_prop_model = best_trial.user_attrs.get("prop_model", None)
            best_preprocess_pipeline = best_trial.user_attrs["preprocess_pipeline"]
            best_preprocess_pipeline.fit(X, y)
            best_model = best_trial.user_attrs["cate_model"]
            if best_metalearner in ["R", "DR", "X", "Z", "U", "F"]:
                X_trans = best_preprocess_pipeline.transform(X)
                best_prop_model.fit(X_trans, t)
                best_prop_est = best_prop_model.predict_proba(X_trans)[:, 1]
                best_prop_est = np.clip(best_prop_est, self.CLIP, 1 - self.CLIP)
                best_model.fit(X_trans, t, y, p=best_prop_est)
            else:
                best_model.fit(best_preprocess_pipeline.transform(X), t, y)

            self.best_prop_models = [best_prop_model]
            self.best_preprocess_pipelines = [best_preprocess_pipeline]
            self.best_models = [best_model]

            print("\tBest trial:\n", best_trial.number)
            print("\tBest prop model:\n", best_prop_model)
            print("\tBest preprocess pipeline:\n", best_preprocess_pipeline)
            print("\tBest model:\n", best_model)
        elif self.ensemble_strategy in ["top5average", "top5distance", "top5ranking"]:
            # Get the top 5 best models, based on the average of the evaluation metrics:
            trial_values = np.array(
                [trial.values if trial.state == optuna.trial.TrialState.COMPLETE else [np.nan] * len(self.evaluators)
                 for trial in self.study.trials])

            if self.visualize:
                plt.boxplot(np.clip(trial_values[~np.isnan(trial_values)], -100, None), labels=self.evaluation_metrics,
                            showfliers=False)
                plt.show()

            # Depending on strategy, calculate euclidian distance of each point to the origin or average the metrics
            if self.ensemble_strategy == "top5distance":
                avg_metrics = np.sqrt(np.sum(trial_values ** 2, axis=1))  # Euclidean norm
            if self.ensemble_strategy == "top5average":
                avg_metrics = np.average(a=trial_values, axis=1)  # Arithmetic mean
            else:  # top5ranking
                rankings = np.argsort(trial_values, axis=0)
                avg_metrics = np.average(a=rankings, axis=1)

            if self.opt_direction == "minimize":
                top5 = np.argsort(avg_metrics)[:self.n_trials] if self.n_trials < 5 else np.argsort(avg_metrics)[:5]
            else:  # self.opt_direction == "maximize":
                top5 = np.argsort(-avg_metrics)[:self.n_trials] if self.n_trials < 5 else np.argsort(-avg_metrics)[:5]

            self.best_preprocess_pipelines, self.best_prop_models, self.best_models = [], [], []
            for trial_number in top5:
                trial = self.study.trials[trial_number]
                preprocess_pipeline = trial.user_attrs["preprocess_pipeline"]
                prop_model = trial.user_attrs.get("prop_model", None)
                cate_model = trial.user_attrs["cate_model"]

                preprocess_pipeline.fit(X, y)
                X_trans = preprocess_pipeline.transform(X)

                if prop_model is not None:
                    prop_model.fit(X_trans, t)
                    prop_est = prop_model.predict_proba(X_trans)[:, 1]
                    prop_est = np.clip(prop_est, self.CLIP, 1 - self.CLIP)
                    cate_model.fit(X_trans, t, y, p=prop_est)
                else:
                    cate_model.fit(X_trans, t, y)

                self.best_preprocess_pipelines.append(preprocess_pipeline)
                self.best_prop_models.append(prop_model)
                self.best_models.append(cate_model)

            print("\tBest prop model:\n", self.best_prop_models)
            print("\tBest preprocess pipeline:\n", self.best_preprocess_pipelines)
            print("\tBest model:\n", self.best_models)
        elif self.ensemble_strategy == "stacking":
            # Get and fit all best models:
            self.best_preprocess_pipelines, self.best_prop_models, self.best_models = [], [], []

            # for trial in self.study.best_trials:
            for trial in self.study.trials:
                preprocess_pipeline = trial.user_attrs["preprocess_pipeline"]
                prop_model = trial.user_attrs.get("prop_model", None)
                cate_model = trial.user_attrs["cate_model"]

                preprocess_pipeline.fit(X, y)
                X_trans = preprocess_pipeline.transform(X)

                if prop_model is not None:
                    prop_model.fit(X_trans, t)
                    prop_est = prop_model.predict_proba(X_trans)[:, 1]
                    prop_est = np.clip(prop_est, self.CLIP, 1 - self.CLIP)
                    cate_model.fit(X_trans, t, y, p=prop_est)
                else:
                    cate_model.fit(X_trans, t, y)

                self.best_preprocess_pipelines.append(preprocess_pipeline)
                self.best_prop_models.append(prop_model)
                self.best_models.append(cate_model)

            # Get model estimates on validation set:
            model_estimates = [[] for _ in range(self.n_folds)]
            for i, (_, val_index) in enumerate(self.cv_splitter()):
                X_val = X[val_index]

                for preprocess_pipeline, prop_model, cate_model in zip(
                        self.best_preprocess_pipelines, self.best_prop_models, self.best_models):
                    X_val_trans = preprocess_pipeline.transform(X_val)
                    if cate_model.__class__.__name__ in ["BaseXRegressor", "BaseXClassifier"]:
                        # For X-Learner, the propensity score is required for prediction
                        prop_est = prop_model.predict_proba(X_val_trans)[:, 1]
                        prop_est = np.clip(prop_est, self.CLIP, 1 - self.CLIP)
                        model_estimates[i].append(cate_model.predict(X_val_trans, p=prop_est)[:, 0])
                    else:
                        model_estimates[i].append(cate_model.predict(X_val_trans)[:, 0])
            model_estimates = np.array(model_estimates)

            # Get evaluator pseudo outcomes:
            evaluator_pseudo_outcomes = [[] for _ in range(self.n_folds)]
            evaluator_weights = [[] for _ in range(self.n_folds)]
            for evaluator_name in self.evaluators.keys():
                for i in range(self.n_folds):
                    pseudo_outcomes = self.evaluators[evaluator_name][i].pseudo_outcomes
                    evaluator_pseudo_outcomes[i].append(pseudo_outcomes)
                    if evaluator_name[0] == "R":
                        # Continue to next evaluator
                        weights = self.evaluators[evaluator_name][i].weights
                        evaluator_weights[i].append(weights)
                    else:
                        evaluator_weights[i].append(np.ones(pseudo_outcomes.size))
            evaluator_pseudo_outcomes = np.array(evaluator_pseudo_outcomes)

            if self.visualize:
                plt.hist(evaluator_pseudo_outcomes.reshape(evaluator_pseudo_outcomes.shape[1],
                                                           evaluator_pseudo_outcomes.shape[2] * self.n_folds))
                plt.legend([key for key in self.evaluators.keys() if key[0] != "R"])
                plt.show()

            # Train linear regression on top of all models:
            reg_alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            weights = []
            for i in range(self.n_folds):
                for j in range(evaluator_pseudo_outcomes.shape[1]):
                    skf = ContStratifiedKFold(n_splits=5)
                    # stacking_regressor = ElasticNetCV(fit_intercept=False, cv=skf, positive=True)
                    stacking_regressor = ConstrainedRegressor()
                    grid_search = GridSearchCV(stacking_regressor, param_grid={'reg_alpha': reg_alphas}, cv=skf,
                                               n_jobs=self.n_jobs)
                    grid_search.fit(X=model_estimates[i, :, :].T, y=evaluator_pseudo_outcomes[i, j, :],
                                    sample_weight=evaluator_weights[i][j])

                    weights.append(grid_search.best_estimator_.coef_)

            self.weights = np.mean(weights, axis=0)

            print("\tEnsemble weights:", np.round(self.weights, 2))
            print('\t', 'Time to build ensemble:', np.round(time.time() - start, 2), 'seconds')

            if self.visualize:
                # Visualize weight distribution:
                plt.pie(self.weights, labels=[f"{i}" for i in range(len(self.weights))])
                plt.show()
        # elif self.ensemble_strategy == "top1constraint":
        #     # Get the single best model, based on the average of the evaluation metrics:
        #     trial_values = np.array(
        #         [trial.values if trial.state == optuna.trial.TrialState.COMPLETE else [np.nan] * len(self.evaluators)
        #          for trial in self.study.trials])
        #
        #     # trial_percentiles = np.zeros_like(trial_values)
        #     # for i in range(trial_values.shape[0]):
        #     #     for j in range(trial_values.shape[1]):
        #     #         trial_percentiles[i, j] = percentileofscore(a=trial_values[:, j], score=trial_values[i, j],
        #     #                                                     kind="weak")
        #
        #     percentiles = np.arange(1, 100, 1)
        #
        #     metric_percentiles = np.percentile(trial_values, q=percentiles, axis=0)
        #
        #     # For each trial, check the percentile rank of each metric:
        #     trial_percentiles = np.zeros(trial_values.shape)
        #     for i in range(trial_values.shape[0]):
        #         if self.opt_direction == "minimize":
        #             trial_percentiles[i] = np.mean(trial_values[i] <= metric_percentiles, axis=0)
        #         else:
        #             trial_percentiles[i] = np.mean(trial_values[i] >= metric_percentiles, axis=0)
        #
        #     # The best trial is the trial with the highest min percentile rank over all metrics:
        #     trial_max_percentiles = np.min(trial_percentiles, axis=1)
        #     best_trial = self.study.trials[np.argmax(trial_max_percentiles)]
        #
        #     # Load best model and refit it on the entire training data
        #     best_metalearner = best_trial.user_attrs["metalearner"]
        #     best_prop_model = best_trial.user_attrs.get("prop_model", None)
        #     best_preprocess_pipeline = best_trial.user_attrs["preprocess_pipeline"]
        #     best_preprocess_pipeline.fit(X, y)
        #     best_model = best_trial.user_attrs["cate_model"]
        #     if best_metalearner in ["R", "DR", "X", "Z", "U", "F"]:
        #         X_trans = best_preprocess_pipeline.transform(X)
        #         best_prop_model.fit(X_trans, t)
        #         best_prop_est = best_prop_model.predict_proba(X_trans)[:, 1]
        #         best_prop_est = np.clip(best_prop_est, self.CLIP, 1 - self.CLIP)
        #         best_model.fit(X_trans, t, y, p=best_prop_est)
        #     else:
        #         best_model.fit(best_preprocess_pipeline.transform(X), t, y)
        #
        #     self.best_prop_models = [best_prop_model]
        #     self.best_preprocess_pipelines = [best_preprocess_pipeline]
        #     self.best_models = [best_model]
        #
        #     print("\tBest trial:\n", best_trial.number)
        #     print("\tBest prop model:\n", best_prop_model)
        #     print("\tBest preprocess pipeline:\n", best_preprocess_pipeline)
        #     print("\tBest model:\n", best_model)
        else:
            raise ValueError(
                "Invalid ensemble strategy. Please choose one of the following: 'pareto', 'top1average', "
                "'top5average', 'top1distance', 'top5distance', 'top1ranking', or 'top5ranking'. "
                "\nFound:", self.ensemble_strategy)

        print('\t', 'Total time to build ensemble:', np.round(time.time() - start, 2), 'seconds')
