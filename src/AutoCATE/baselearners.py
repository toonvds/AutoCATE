from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_base_learner(trial, task="regression", base_learner="RF", meta_learner="", joint_optimization=False,
                     n_jobs=None):
    if base_learner == "RF":
        return get_random_forest(trial, task=task, meta_learner=meta_learner, joint_optimization=joint_optimization,
                                 n_jobs=n_jobs)
    elif base_learner == "LR":
        return get_logistic_regression(trial, task=task, meta_learner=meta_learner,
                                       joint_optimization=joint_optimization, n_jobs=n_jobs)
    elif base_learner == "GB":
        return get_gradient_boosting(trial, task=task, meta_learner=meta_learner, joint_optimization=joint_optimization,
                                     n_jobs=n_jobs)
    elif base_learner == "ET":
        return get_extra_trees(trial, task=task, meta_learner=meta_learner, joint_optimization=joint_optimization,
                               n_jobs=n_jobs)
    elif base_learner == "GP":
        return get_gaussian_process(trial, task=task, meta_learner=meta_learner, joint_optimization=joint_optimization,
                                    n_jobs=n_jobs)
    elif base_learner == "SVM":
        return get_support_vector_machine(trial, task=task, meta_learner=meta_learner,
                                          joint_optimization=joint_optimization, n_jobs=n_jobs)
    elif base_learner == "kNN":
        return get_k_nearest_neighbors(trial, task=task, meta_learner=meta_learner,
                                       joint_optimization=joint_optimization, n_jobs=n_jobs)
    elif base_learner == "NN":
        return get_neural_network(trial, task=task, meta_learner=meta_learner, joint_optimization=joint_optimization,
                                  n_jobs=n_jobs)
    elif base_learner == "DT":
        return get_decision_tree(trial, task=task, meta_learner=meta_learner, joint_optimization=joint_optimization,
                                 n_jobs=n_jobs)
    else:
        raise ValueError("Base learner not implemented")


def get_random_forest(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    rf_args = {
        "n_estimators": trial.suggest_int(prefix + "rf_n_estimators", 50, 500),
        "max_depth": None,
        "min_samples_split": trial.suggest_int(prefix + "rf_min_samples_split", 2, 100),
        "max_features": trial.suggest_float(prefix + "rf_max_features", 0.4, 1.0),
        "n_jobs": n_jobs,
    }

    if task == "regression":
        if not joint_optimization:
            rf_args.update(
                {"criterion": trial.suggest_categorical(prefix + "rf_criterion", ["squared_error", "absolute_error"])})
        return RandomForestRegressor(**rf_args)
    elif task == "classification":
        if not joint_optimization:
            rf_args.update({"criterion": trial.suggest_categorical(prefix + "rf_criterion", ["gini", "entropy"])})
        return RandomForestClassifier(**rf_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_logistic_regression(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    lr_args = {
        "alpha" if task == "regression" else "C": trial.suggest_float(prefix + "lr_alpha", 1e-6, 1e6, log=True),
    }

    if task == "regression":
        return Ridge(**lr_args)
    elif task == "classification":
        return LogisticRegression(**lr_args, max_iter=20000, n_jobs=n_jobs)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_gradient_boosting(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    gb_args = {
        "n_estimators": trial.suggest_int(prefix + "gb_n_estimators", 50, 2000, log=True),
        "subsample": trial.suggest_float(prefix + "gb_subsample", 0.4, 1.0),
        "min_samples_split": trial.suggest_int(prefix + "gb_min_samples_split", 2, 500),
        "learning_rate": trial.suggest_float(prefix + "gb_learning_rate", 0.05, 0.5),
        "n_iter_no_change": trial.suggest_int(prefix + "gb_n_iter_no_change", 5, 100),
        "max_leaf_nodes": None,
        "max_depth": None,
    }

    if task == "regression":
        return GradientBoostingRegressor(**gb_args)
    elif task == "classification":
        return GradientBoostingClassifier(**gb_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_extra_trees(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    et_args = {
        "n_estimators": trial.suggest_int(prefix + "et_n_estimators", 50, 500),
        "max_depth": None,
        # "max_depth": trial.suggest_int(prefix + "et_max_depth", 1, 1000, log=True),
        "min_samples_split": trial.suggest_int(prefix + "et_min_samples_split", 2, 100),
        "max_features": trial.suggest_float(prefix + "et_max_features", 0.4, 1.0),
        # "max_samples": trial.suggest_float(prefix + "et_max_samples", 0.4, 1.0),
        "n_jobs": n_jobs,
    }

    if task == "regression":
        return ExtraTreesRegressor(**et_args)
    elif task == "classification":
        return ExtraTreesClassifier(**et_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_gaussian_process(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    gp_args = {
        "n_restarts_optimizer": trial.suggest_int(prefix + "gp_n_restarts_optimizer", 0, 5),
    }

    if task == "regression":
        gp_args.update({
            "normalize_y": trial.suggest_categorical(prefix + "gp_normalize_y", [True, False]),
            "alpha": trial.suggest_float(prefix + "gp_alpha", 1e-5, 1e-2, log=True),
        })
        return GaussianProcessRegressor(**gp_args)
    elif task == "classification":
        gp_args.update({
            "max_iter_predict": trial.suggest_int(prefix + "gp_max_iter_predict", 100, 1000),
        })
        return GaussianProcessClassifier(**gp_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_support_vector_machine(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    svm_args = {
        "C": trial.suggest_float(prefix + "svm_C", 1e-6, 1e6, log=True),
        "kernel": trial.suggest_categorical(prefix + "svm_kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "degree": trial.suggest_int(prefix + "svm_degree", 1, 10),
    }

    # Set max_iter to not get stuck on one trial
    trial.set_user_attr("svm_max_iter", 2000)
    svm_args.update({"max_iter": trial.user_attrs["svm_max_iter"]})

    if task == "regression":
        return SVR(**svm_args)
    elif task == "classification":
        return SVC(**svm_args, probability=True)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_k_nearest_neighbors(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    knn_args = {
        "n_neighbors": trial.suggest_int(prefix + "knn_n_neighbors", 1, 30),
        "weights": trial.suggest_categorical(prefix + "knn_weights", ["uniform", "distance"]),
        "n_jobs": n_jobs,
    }

    if task == "regression":
        return KNeighborsRegressor(**knn_args)
    elif task == "classification":
        return KNeighborsClassifier(**knn_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_neural_network(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    hidden_layers = trial.suggest_int(prefix + "nn_hidden_layers", 1, 3)
    hidden_neurons = trial.suggest_int(prefix + "nn_hidden_neurons", 8, 64)
    nn_args = {"hidden_layer_sizes": [hidden_neurons] * hidden_layers,
               "alpha": trial.suggest_float(prefix + "nn_alpha", 1e-6, 1e1, log=True),
               # "solver": trial.suggest_categorical(prefix + "nn_solver", ["lbfgs", "adam"]),
               "learning_rate_init": trial.suggest_float(prefix + "nn_learning_rate_init", 5e-4, 1e-2, log=True),
               "batch_size": trial.suggest_int(prefix + "nn_batch_size", 16, 64),
               "activation": trial.suggest_categorical(prefix + "nn_activation", ["tanh", "relu"]),
               }

    trial.set_user_attr("nn_max_iter", 200)
    nn_args.update({"max_iter": trial.user_attrs["nn_max_iter"]})
    trial.set_user_attr("nn_solver", "adam")
    nn_args.update({"solver": trial.user_attrs["nn_solver"]})
    trial.set_user_attr("nn_early_stopping", True)
    nn_args.update({"early_stopping": trial.user_attrs["nn_early_stopping"]})
    # trial.set_user_attr("nn_n_iter_no_change", 20)
    # nn_args.update({"n_iter_no_change": trial.user_attrs["nn_n_iter_no_change"]})

    if task == "regression":
        return MLPRegressor(**nn_args)
    elif task == "classification":
        return MLPClassifier(**nn_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_decision_tree(trial, task="regression", meta_learner="", joint_optimization=False, n_jobs=None):
    prefix = "" if joint_optimization else str(meta_learner) + "_"

    dt_args = {
        "max_depth": trial.suggest_int(prefix + "dt_max_depth", 1, 2000, log=True),
        "min_samples_split": trial.suggest_int(prefix + "dt_min_samples_split", 2, 500),
        "min_samples_leaf": trial.suggest_int(prefix + "dt_min_samples_leaf", 1, 500),
        "max_features": trial.suggest_float(prefix + "dt_max_features", 0.4, 1.0),
    }

    if task == "regression":
        return DecisionTreeRegressor(**dt_args)
    elif task == "classification":
        return DecisionTreeClassifier(**dt_args)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")
