"""
This script is used to test and benchmark the AutoCATE algorithm on various data sets.
Author: Toon Vanderschueren
"""

import time
import argparse
from data.utils import load_hillstrom, load_information

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

from src.AutoCATE.AutoCATE import AutoCATE
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
from src.AutoCATE.utils import auc_qini, SingleSplit


# Read arguments from parser
parser = argparse.ArgumentParser(description='Run AutoCATE on various data sets.')
parser.add_argument('--dataset', type=str, default='Information', help='Dataset to run AutoCATE on.')
parser.add_argument('--experiment_iter', type=int, default=1, help='Number of experiments to run.')
parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for the cross-validation.')
parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for the hyperparameter optimization.')
parser.add_argument('--n_eval_versions', type=int, default=1, help='Number of evaluation versions.')
parser.add_argument('--n_eval_trials', type=int, default=5, help='Number of trials for the evaluation.')
parser.add_argument('--ensemble', type=str, default="top1average", help='Ensemble method to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--AutoCATE', action=argparse.BooleanOptionalAction, help='Run AutoCATE.')
parser.set_defaults(AutoCATE=True)
parser.add_argument('--eval_metrics', '--names-list', default=["T"], nargs='+',
                    help='The evaluation measures to use.')  # ["kNN", "R", "DR", "Z", "U", "F", "T"]  # ["kNN", "R", "DR", "Z", "U", "F", "T", "IF"]
parser.add_argument('--holdout_ratio', type=float, default=0.3, help='Ratio of data to use for validation.')
parser.add_argument('--s_random_forest', action=argparse.BooleanOptionalAction,
                    help='Run Random Forest S-Learner.')
parser.set_defaults(s_random_forest=False)
parser.add_argument('--s_gradient_boosting', action=argparse.BooleanOptionalAction,
                    help='Run Gradient Boosting S-Learner.')
parser.set_defaults(s_gradient_boosting=False)
parser.add_argument('--s_linear_regression', action=argparse.BooleanOptionalAction,
                    help='Run Linear Regression S-Learner.')
parser.set_defaults(s_linear_regression=False)
parser.add_argument('--t_random_forest', action=argparse.BooleanOptionalAction,
                    help='Run Random Forest T-Learner.')
parser.set_defaults(t_random_forest=True)
parser.add_argument('--t_gradient_boosting', action=argparse.BooleanOptionalAction,
                    help='Run Gradient Boosting T-Learner.')
parser.set_defaults(t_gradient_boosting=True)
parser.add_argument('--t_linear_regression', action=argparse.BooleanOptionalAction,
                    help='Run Linear Regression T-Learner.')
parser.set_defaults(t_linear_regression=False)
args = parser.parse_args()

np.random.seed(42)

# Create the arrays to store all the results
if args.AutoCATE:
    auqcs = []

if args.s_random_forest:
    auqcs_srf = []

if args.s_gradient_boosting:
    auqcs_sgb = []

if args.s_linear_regression:
    auqcs_slr = []

if args.t_random_forest:
    auqcs_trf = []

if args.t_gradient_boosting:
    auqcs_tgb = []

if args.t_linear_regression:
    auqcs_tlr = []

# Define the search space for Random Forest and Gradient Boosting
rf_search_space = {
    "n_estimators": [int(x) for x in np.linspace(start=50, stop=500, num=50)],
    "min_samples_split": [int(x) for x in np.linspace(start=2, stop=100, num=50)],
    "max_features": [float(x) for x in np.linspace(start=0.4, stop=1., num=50)],
    "criterion": ["squared_error", "absolute_error"],
}

gb_search_space = {
    "n_estimators": [int(x) for x in np.linspace(start=50, stop=2000, num=50)],
    "subsample": [float(x) for x in np.linspace(start=0.4, stop=1.0, num=50)],
    "min_samples_split": [int(x) for x in np.linspace(start=2, stop=500, num=50)],
    "learning_rate": [float(x) for x in np.linspace(start=0.05, stop=0.5, num=50)],
    "n_iter_no_change": [int(x) for x in np.linspace(start=5, stop=100, num=50)],
}

start = time.time()

for dataset_iter in range(args.experiment_iter):  # For testing purposes
    print('\nIteration: ', dataset_iter + 1)

    # Load the data
    if args.dataset == 'Hillstrom':
        X, t, yf = load_hillstrom()
        X_train, X_test, t_train, t_test, yf_train, yf_test = train_test_split(X, t, yf, test_size=0.3,
                                                                               random_state=42)
    elif args.dataset == 'Information':
        X, t, yf = load_information()
        X_train, X_test, t_train, t_test, yf_train, yf_test = train_test_split(X, t, yf, test_size=0.3,
                                                                               random_state=42)
    else:
        raise ValueError("Dataset not implemented.")

    # We will calculate the AUQC using the Z-Learner's pseudo-outcome
    z_test = np.zeros_like(yf_test)
    z_test[t_test == 1] = yf_test[t_test == 1] / t_test.mean()
    z_test[t_test == 0] = - yf_test[t_test == 0] / (1 - t_test.mean())

    print('Uplift data assumed to be classification--unique values =', np.unique(yf_train).shape[0])
    task = "classification" # if args.dataset == "" else "regression"

    if task == "classification":
        rf_search_space["criterion"] = ["gini", "entropy"]

    # Run AutoCATE
    if args.AutoCATE:
        autocate = AutoCATE(n_folds=args.n_folds, n_trials=args.n_trials, n_eval_versions=args.n_eval_versions,
                            n_eval_trials=args.n_eval_trials, ensemble_strategy=args.ensemble,
                            evaluation_metrics=args.eval_metrics, seed=args.seed,
                            task=task, holdout_ratio=args.holdout_ratio, n_jobs=-1)
        autocate.fit(X_train, t_train, yf_train)
        ite_pred = autocate.predict(X_test)

        auqc = auc_qini(z_test, ite_pred)
        auqcs.append(auqc)
        print('AutoCATE AUQC:\t', auqc)

    # Compare with RF S-Learner
    if args.s_random_forest:
        rf = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        skf = SingleSplit(test_size=0.3) if args.n_folds == 1 else args.n_folds
        rf_tuned = RandomizedSearchCV(estimator=rf, param_distributions=rf_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        rf_tuned.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
        if task == "classification":
            ite_pred = rf_tuned.predict_proba(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1))[:, 1] - \
                       rf_tuned.predict_proba(np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))[:, 1]
        else:
            ite_pred = rf_tuned.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - \
                       rf_tuned.predict(np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

        auqc = auc_qini(z_test, ite_pred)
        auqcs_srf.append(auqc)
        print('SRF AUQC:\t', auqc)

    # Compare with RF T-Learner
    if args.t_random_forest:
        rf0 = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        rf1 = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        skf = SingleSplit(test_size=0.3) if args.n_folds == 1 else args.n_folds

        rf0_tuned = RandomizedSearchCV(estimator=rf0, param_distributions=rf_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        rf0_tuned.fit(X_train[t_train == 0], yf_train[t_train == 0])
        rf1_tuned = RandomizedSearchCV(estimator=rf1, param_distributions=rf_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        rf1_tuned.fit(X_train[t_train == 1], yf_train[t_train == 1])

        if task == "classification":
            ite_pred = rf1_tuned.predict_proba(X_test)[:, 1] - rf0_tuned.predict_proba(X_test)[:, 1]
        else:
            ite_pred = rf1_tuned.predict(X_test) - rf0_tuned.predict(X_test)

        auqc = auc_qini(z_test, ite_pred)
        auqcs_trf.append(auqc)
        print('TRF AUQC:\t', auqc)

    # Compare with GB S-Learner
    if args.s_gradient_boosting:
        gb = GradientBoostingRegressor() if task == "regression" else GradientBoostingClassifier()
        skf = SingleSplit(test_size=0.3) if args.n_folds == 1 else args.n_folds
        gb_tuned = RandomizedSearchCV(estimator=gb, param_distributions=gb_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        gb_tuned.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)

        if task == "classification":
            ite_pred = gb_tuned.predict_proba(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1))[:, 1] - \
                       gb_tuned.predict_proba(np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))[:, 1]
        else:
            ite_pred = gb_tuned.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - \
                       gb_tuned.predict(np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

        auqc = auc_qini(z_test, ite_pred)
        auqcs_sgb.append(auqc)
        print('SGB AUQC:\t', auqc)

    # Compare with GB T-Learner
    if args.t_gradient_boosting:
        gb0 = GradientBoostingRegressor() if task == "regression" else GradientBoostingClassifier()
        gb1 = GradientBoostingRegressor() if task == "regression" else GradientBoostingClassifier()
        skf = SingleSplit(test_size=0.3) if args.n_folds == 1 else args.n_folds

        gb0_tuned = RandomizedSearchCV(estimator=gb0, param_distributions=gb_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        gb0_tuned.fit(X_train[t_train == 0], yf_train[t_train == 0])
        gb1_tuned = RandomizedSearchCV(estimator=gb1, param_distributions=gb_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        gb1_tuned.fit(X_train[t_train == 1], yf_train[t_train == 1])

        if task == "classification":
            ite_pred = gb1_tuned.predict_proba(X_test)[:, 1] - gb0_tuned.predict_proba(X_test)[:, 1]
        else:
            ite_pred = gb1_tuned.predict(X_test) - gb0_tuned.predict(X_test)

        auqc = auc_qini(z_test, ite_pred)
        auqcs_tgb.append(auqc)
        print('TGB AUQC:\t', auqc)

    # Compare with LR S-Learner
    if args.s_linear_regression:
        lr = LinearRegression() if task == "regression" else LogisticRegression(penalty=None)
        lr.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)

        if task == "classification":
            ite_pred = lr.predict_proba(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1))[:, 1] - \
                       lr.predict_proba(np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))[:, 1]
        else:
            ite_pred = lr.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - \
                       lr.predict(np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

        auqc = auc_qini(z_test, ite_pred)
        auqcs_slr.append(auqc)
        print('SLR AUQC:\t', auqc)

    # Compare with LR T-Learner
    if args.t_linear_regression:
        lr0 = LinearRegression() if task == "regression" else LogisticRegression(penalty=None)
        lr1 = LinearRegression() if task == "regression" else LogisticRegression(penalty=None)

        lr0.fit(X_train[t_train == 0], yf_train[t_train == 0])
        lr1.fit(X_train[t_train == 1], yf_train[t_train == 1])

        if task == "classification":
            ite_pred = lr1.predict_proba(X_test)[:, 1] - lr0.predict_proba(X_test)[:, 1]
        else:
            ite_pred = lr1.predict(X_test) - lr0.predict(X_test)

        auqc = auc_qini(z_test, ite_pred)
        auqcs_tlr.append(auqc)
        print('TLR AUQC:\t', auqc)

print("\n\nRESULTS")
if args.AutoCATE:
    print("AutoCATE")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs), 4), " | SE: ", np.round(sem(auqcs), 4))
if args.s_random_forest:
    print("RF S-Learner")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_srf), 4), " | SE: ", np.round(sem(auqcs_srf), 4))
if args.t_random_forest:
    print("RF T-Learner")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_trf), 4), " | SE: ", np.round(sem(auqcs_trf), 4))
if args.s_gradient_boosting:
    print("GB S-Learner")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_sgb), 4), " | SE: ", np.round(sem(auqcs_sgb), 4))
if args.t_gradient_boosting:
    print("GB T-Learner")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_tgb), 4), " | SE: ", np.round(sem(auqcs_tgb), 4))
if args.s_linear_regression:
    print("LR S-Learner")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_slr), 4), " | SE: ", np.round(sem(auqcs_slr), 4))
if args.t_linear_regression:
    print("LR T-Learner")
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_tlr), 4), " | SE: ", np.round(sem(auqcs_tlr), 4))

print('\n\nTime elapsed: ', np.round(time.time() - start, 2))

# Write the results to a txt file:
title = "experiments/results/results_"
title += str(args.dataset) + '_' + str(args.experiment_iter) + '_iter_'
if args.AutoCATE:
    title += "AutoCATE_"
if args.s_random_forest:
    title += "SRF_"
if args.t_random_forest:
    title += "TRF_"
if args.s_gradient_boosting:
    title += "SGB_"
if args.t_gradient_boosting:
    title += "TGB_"
if args.s_linear_regression:
    title += "SLR_"
if args.t_linear_regression:
    title += "TLR_"
title += str(args.n_folds) + '_folds_' + str(args.n_trials) + '_trials_'
if args.AutoCATE:
    title += str(args.n_eval_versions) + '_eval_versions_' + str(args.n_eval_trials) + '_eval_trials_' + str(
        args.ensemble) + '_ensemble_' + str(args.holdout_ratio) + '_holdout_'
    title += '_'.join(args.eval_metrics) + '_metrics_'
    title += '_'.join(autocate.metalearners) + '_metalearners_'
    title += '_'.join(autocate.base_learners) + '_baselearners'

with open(title + '.txt', 'w') as f:
    f.write("\n\nRESULTS\n")
    if args.AutoCATE:
        f.write("\n\nAutoCATE\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs) + " \\pm " + "%.2f" % sem(auqcs) + "\n")
    if args.s_random_forest:
        f.write("\n\nRF S-Learner\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_srf) + " \\pm " + "%.2f" % sem(auqcs_srf) + "\n")
    if args.t_random_forest:
        f.write("\n\nRF T-Learner\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_trf) + " \\pm " + "%.2f" % sem(auqcs_trf) + "\n")
    if args.s_gradient_boosting:
        f.write("\n\nGB S-Learner\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_sgb) + " \\pm " + "%.2f" % sem(auqcs_sgb) + "\n")
    if args.t_gradient_boosting:
        f.write("\n\nGB T-Learner\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_tgb) + " \\pm " + "%.2f" % sem(auqcs_tgb) + "\n")
    if args.s_linear_regression:
        f.write("\n\nLR S-Learner\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_slr) + " \\pm " + "%.2f" % sem(auqcs_slr) + "\n")
    if args.t_linear_regression:
        f.write("\n\nLR T-Learner\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_tlr) + " \\pm " + "%.2f" % sem(auqcs_tlr) + "\n")

    f.write("\n\nTime elapsed: " + "%.2f" % (time.time() - start) + "\n")

    f.write("\n" + str(args))
    f.write("\n______________________________________________________________________________________\n")
