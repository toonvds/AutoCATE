"""
This script is used to test and benchmark the AutoCATE algorithm on various data sets.
Author: Toon Vanderschueren
"""

import time
import argparse
from data.utils import load_ihdp_iteration, load_twins, load_acic_iteration, load_news_iteration

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt

from src.AutoCATE.AutoCATE import AutoCATE
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,
                              GradientBoostingClassifier)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
from src.AutoCATE.utils import auc_qini, SingleSplit

# Read arguments from parser
parser = argparse.ArgumentParser(description='Run AutoCATE on various data sets.')
parser.add_argument('--dataset', type=str, default='ACIC', help='Dataset to run AutoCATE on.')
parser.add_argument('--experiment_iter', type=int, default=4, help='Number of experiments to run.')
parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for the cross-validation.')
parser.add_argument('--n_trials', type=int, default=30, help='Number of trials for the hyperparameter optimization.')
parser.add_argument('--n_eval_versions', type=int, default=1, help='Number of evaluation versions.')
parser.add_argument('--n_eval_trials', type=int, default=30, help='Number of trials for the evaluation.')
parser.add_argument('--ensemble', type=str, default="top1", help='Ensemble method to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--AutoCATE', action=argparse.BooleanOptionalAction, help='Run AutoCATE.')
parser.set_defaults(AutoCATE=True)
parser.add_argument('--random_forest', action=argparse.BooleanOptionalAction, help='Run Random Forest.')
parser.set_defaults(random_forest=False)
parser.add_argument('--gradient_boosting', action=argparse.BooleanOptionalAction, help='Run Gradient Boosting.')
parser.set_defaults(gradient_boosting=False)
parser.add_argument('--linear_regression', action=argparse.BooleanOptionalAction, help='Run Linear Regression.')
parser.set_defaults(linear_regression=False)
args = parser.parse_args()

np.random.seed(42)

# Create the arrays to store all the results
if args.AutoCATE:
    pehes = []
    mapes = []
    r2s = []
    auqcs = []

if args.random_forest:
    pehes_rf = []
    mapes_rf = []
    r2s_rf = []
    auqcs_rf = []

if args.gradient_boosting:
    pehes_gb = []
    mapes_gb = []
    r2s_gb = []
    auqcs_gb = []

if args.linear_regression:
    pehes_lr = []
    mapes_lr = []
    r2s_lr = []
    auqcs_lr = []

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
    if args.dataset == 'IHDP':
        X_train, t_train, yf_train, X_test, _, _, ite_test = load_ihdp_iteration(dataset_iter)
    elif args.dataset == 'Twins':
        X, t, yf, ite = load_twins()
        X_train, X_test, t_train, _, yf_train, _, _, ite_test = train_test_split(X, t, yf, ite, test_size=0.3,
                                                                                 random_state=42)
    elif args.dataset == 'ACIC':
        X, t, yf, ite = load_acic_iteration(dataset_iter)
        X_train, X_test, t_train, _, yf_train, _, _, ite_test = train_test_split(X, t, yf, ite, test_size=0.3,
                                                                                 random_state=42)
    elif args.dataset == 'News':
        X, t, yf, ite = load_news_iteration(dataset_iter)
        X_train, X_test, t_train, _, yf_train, _, _, ite_test = train_test_split(X, t, yf, ite, test_size=0.3,
                                                                                 random_state=42)
    else:
        raise ValueError("Dataset not implemented.")

    task = "classification" if args.dataset == "Twins" else "regression"

    # Run AutoCATE
    if args.AutoCATE:
        autocate = AutoCATE(n_folds=args.n_folds, n_trials=args.n_trials, n_eval_versions=args.n_eval_versions,
                            n_eval_trials=args.n_eval_trials, ensemble_strategy=args.ensemble, seed=args.seed,
                            task=task)
        autocate.fit(X_train, t_train, yf_train)
        ite_pred = autocate.predict(X_test)

        pehe = root_mean_squared_error(ite_test, ite_pred)
        mape = mean_absolute_percentage_error(ite_test, ite_pred)
        r2 = r2_score(ite_test, ite_pred)
        auqc = auc_qini(ite_test, ite_pred)

        fig, ax = plt.subplots()
        ax.set_title(str(args.dataset) + '(iter: ' + str(dataset_iter + 1) + '; sqPEHE: ' + str(round(pehe, 2)) + ')')
        ax.plot(ite_test, ite_pred, 'o', linestyle='None', color='green')
        # Add a linear trend line
        ax.plot(np.unique(ite_test), np.poly1d(np.polyfit(ite_test, ite_pred, 1))(np.unique(ite_test)),
                alpha=0.1, color='green')
        # Add a x=y line
        ax.plot(np.unique(ite_test), np.unique(ite_test), 'b', linestyle='--', alpha=0.2)
        plt.show()

        pehes.append(pehe)
        mapes.append(mape)
        r2s.append(r2)
        auqcs.append(auqc)
        print('AutoCATE PEHE:\t', pehe)
        print('AutoCATE MAPE:\t', mape)
        print('AutoCATE R2:  \t', r2)
        print('AutoCATE AUQC:\t', auqc)

    # Compare with RF S-Learner
    if args.random_forest:
        rf = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
        skf = SingleSplit(test_size=0.3) if args.n_folds == 1 else args.n_folds
        rf_tuned = RandomizedSearchCV(estimator=rf, param_distributions=rf_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        rf_tuned.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
        ite_pred = rf_tuned.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - rf_tuned.predict(
            np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

        pehe = root_mean_squared_error(ite_test, ite_pred)
        mape = mean_absolute_percentage_error(ite_test, ite_pred)
        r2 = r2_score(ite_test, ite_pred)
        auqc = auc_qini(ite_test, ite_pred)

        pehes_rf.append(pehe)
        mapes_rf.append(mape)
        r2s_rf.append(r2)
        auqcs_rf.append(auqc)
        print('\nRF PEHE:\t', pehe)
        print('RF MAPE:\t', mape)
        print('RF R2:  \t', r2)
        print('RF AUQC:\t', auqc)

    # Compare with GB S-Learner
    if args.gradient_boosting:
        gb = GradientBoostingRegressor() if task == "regression" else GradientBoostingClassifier()
        skf = SingleSplit(test_size=0.3) if args.n_folds == 1 else args.n_folds
        gb_tuned = RandomizedSearchCV(estimator=gb, param_distributions=gb_search_space, n_iter=args.n_trials, cv=skf,
                                      random_state=42, n_jobs=-1, verbose=0)
        gb_tuned.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
        ite_pred = gb_tuned.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - gb_tuned.predict(
            np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

        pehe = root_mean_squared_error(ite_test, ite_pred)
        mape = mean_absolute_percentage_error(ite_test, ite_pred)
        r2 = r2_score(ite_test, ite_pred)
        auqc = auc_qini(ite_test, ite_pred)

        pehes_gb.append(pehe)
        mapes_gb.append(mape)
        r2s_gb.append(r2)
        auqcs_gb.append(auqc)
        print('\nGB PEHE:\t', pehe)
        print('GB MAPE:\t', mape)
        print('GB R2:  \t', r2)
        print('GB AUQC:\t', auqc)

    # Compare with LR S-Learner
    if args.linear_regression:
        lr = LinearRegression()
        lr.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
        ite_pred = lr.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - lr.predict(
            np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

        # pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))
        pehe = root_mean_squared_error(ite_test, ite_pred)
        mape = mean_absolute_percentage_error(ite_test, ite_pred)
        r2 = r2_score(ite_test, ite_pred)
        auqc = auc_qini(ite_test, ite_pred)

        pehes_lr.append(pehe)
        mapes_lr.append(mape)
        r2s_lr.append(r2)
        auqcs_lr.append(auqc)
        print('\nLR PEHE:\t', pehe)
        print('LR MAPE:\t', mape)
        print('LR R2:  \t', r2)
        print('LR AUQC:\t', auqc)

print("\n\nRESULTS")
if args.AutoCATE:
    print("AutoCATE")
    print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes), 4), " | SE: ", np.round(sem(pehes), 4))
    print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes), 4), " | SE: ", np.round(sem(mapes), 4))
    print("\tR2 -- Avg:        \t", np.round(np.mean(r2s), 4), " | SE: ", np.round(sem(r2s), 4))
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs), 4), " | SE: ", np.round(sem(auqcs), 4))
if args.random_forest:
    print("RF S-Learner")
    print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_rf), 4), " | SE: ", np.round(sem(pehes_rf), 4))
    print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_rf), 4), " | SE: ", np.round(sem(mapes_rf), 4))
    print("\tR2 -- Avg:        \t", np.round(np.mean(r2s_rf), 4), " | SE: ", np.round(sem(r2s_rf), 4))
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_rf), 4), " | SE: ", np.round(sem(auqcs_rf), 4))
if args.gradient_boosting:
    print("GB S-Learner")
    print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_gb), 4), " | SE: ", np.round(sem(pehes_gb), 4))
    print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_gb), 4), " | SE: ", np.round(sem(mapes_gb), 4))
    print("\tR2 -- Avg:        \t", np.round(np.mean(r2s_gb), 4), " | SE: ", np.round(sem(r2s_gb), 4))
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_gb), 4), " | SE: ", np.round(sem(auqcs_gb), 4))
if args.linear_regression:
    print("LR S-Learner")
    print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_lr), 4), " | SE: ", np.round(sem(pehes_lr), 4))
    print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_lr), 4), " | SE: ", np.round(sem(mapes_lr), 4))
    print("\tR2 -- Avg:        \t", np.round(np.mean(r2s_lr), 4), " | SE: ", np.round(sem(r2s_lr), 4))
    print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_lr), 4), " | SE: ", np.round(sem(auqcs_lr), 4))

# fig, ax = plt.subplots()
# ax.set_title('IHDP - PEHE')
# ax.boxplot([pehes, pehes_rf, pehes_gb, pehes_lr], labels=['AutoCATE', 'RF S-Learner', 'GB S-Learner', 'LR S-Learner'])
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots()
# ax.set_title('IHDP - MAPE')
# ax.boxplot([mapes, mapes_rf, mapes_gb, mapes_lr], labels=['AutoCATE', 'RF S-Learner', 'GB S-Learner', 'LR S-Learner'])
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots()
# ax.set_title('IHDP - R2')
# ax.boxplot([r2s, r2s_rf, r2s_gb, r2s_lr], labels=['AutoCATE', 'RF S-Learner', 'GB S-Learner', 'LR S-Learner'])
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots()
# ax.set_title('IHDP - AUQC')
# ax.boxplot([auqcs, auqcs_rf, auqcs_gb, auqcs_lr], labels=['AutoCATE', 'RF S-Learner', 'GB S-Learner', 'LR S-Learner'])
# plt.tight_layout()
# plt.show()

print('\n\nTime elapsed: ', np.round(time.time() - start, 2))

# Write the results to a txt file:
title = "experiments/results/results_"
title += str(args.dataset) + '_' + str(args.experiment_iter) + '_iter_'
if args.AutoCATE:
    title += "AutoCATE"
if args.random_forest:
    title += "RF"
if args.gradient_boosting:
    title += "GB"
if args.linear_regression:
    title += "LR"
title += '_' + str(args.n_folds) + '_folds_' + str(args.n_trials) + '_trials_'
if args.AutoCATE:
    title += str(args.n_eval_versions) + '_eval_versions_' + str(args.n_eval_trials) + '_eval_trials' + str(args.ensemble) + '_ensemble'

with open(title + '.txt', 'w') as f:
    f.write("\n\nRESULTS\n")
    if args.AutoCATE:
        f.write("\n\nAutoCATE\n")
        f.write("\tsqrt PEHE: \t" + "%.2f" % np.mean(pehes) + " \\pm " + "%.2f" % sem(pehes) + "\n")
        f.write("\tMAPE:      \t" + "%.2f" % np.mean(mapes) + " \\pm " + "%.2f" % sem(mapes) + "\n")
        f.write("\tR2:        \t" + "%.2f" % np.mean(r2s) + " \\pm " + "%.2f" % sem(r2s) + "\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs) + " \\pm " + "%.2f" % sem(auqcs) + "\n")
    if args.random_forest:
        f.write("\n\nRF S-Learner\n")
        f.write("\tsqrt PEHE: \t" + "%.2f" % np.mean(pehes_rf) + " \\pm " + "%.2f" % sem(pehes_rf) + "\n")
        f.write("\tMAPE:      \t" + "%.2f" % np.mean(mapes_rf) + " \\pm " + "%.2f" % sem(mapes_rf) + "\n")
        f.write("\tR2:        \t" + "%.2f" % np.mean(r2s_rf) + " \\pm " + "%.2f" % sem(r2s_rf) + "\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_rf) + " \\pm " + "%.2f" % sem(auqcs_rf) + "\n")
    if args.gradient_boosting:
        f.write("\n\nGB S-Learner\n")
        f.write("\tsqrt PEHE: \t" + "%.2f" % np.mean(pehes_gb) + " \\pm " + "%.2f" % sem(pehes_gb) + "\n")
        f.write("\tMAPE:      \t" + "%.2f" % np.mean(mapes_gb) + " \\pm " + "%.2f" % sem(mapes_gb) + "\n")
        f.write(
            "\tR2:        \t" + "%.2f" % np.mean(r2s_gb) + " \\pm " + "%.2f" % sem(r2s_gb) + "\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_gb) + " \\pm " + "%.2f" % sem(auqcs_gb) + "\n")
    if args.linear_regression:
        f.write("\n\nLR S-Learner\n")
        f.write("\tsqrt PEHE: \t" + "%.2f" % np.mean(pehes_lr) + " \\pm " + "%.2f" % sem(pehes_lr) + "\n")
        f.write("\tMAPE:      \t" + "%.2f" % np.mean(mapes_lr) + " \\pm " + "%.2f" % sem(mapes_lr) + "\n")
        f.write("\tR2:        \t" + "%.2f" % np.mean(r2s_lr) + " \\pm " + "%.2f" % sem(r2s_lr) + "\n")
        f.write("\tAUQC:      \t" + "%.2f" % np.mean(auqcs_lr) + " \\pm " + "%.2f" % sem(auqcs_lr) + "\n")

    f.write("\n\nTime elapsed: " + "%.2f" % (time.time() - start) + "\n")

    f.write("\n" + str(args))
    f.write("\n______________________________________________________________________________________\n")

