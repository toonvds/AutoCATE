"""
This script is used to test and benchmark the AutoCATE algorithm on the data from the ACIC 2016 competition.
Author: Toon Vanderschueren
"""

import time
from data.utils import load_acic_iteration

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

from src.AutoCATE.AutoCATE import AutoCATE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
from src.AutoCATE.utils import auc_qini
from sklearn.model_selection import train_test_split, RandomizedSearchCV


np.random.seed(42)

pehes = []
pehes_rf = []
pehes_lr = []

mapes = []
mapes_rf = []
mapes_lr = []

r2s = []
r2s_rf = []
r2s_lr = []

auqcs = []
auqcs_rf = []
auqcs_lr = []

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

# Iterate over all 77 iterations of the ACIC dataset:
# for dataset_iter in range(77):
for dataset_iter in range(10, 13):  # For testing purposes
    print('Iteration: ', dataset_iter + 1)
    X, t, yf, ite = load_acic_iteration(dataset_iter)

    X_train, X_test, t_train, _, yf_train, _, _, ite_test = train_test_split(X, t, yf, ite, test_size=0.3,
                                                                             random_state=42)

    # AutoCATE
    autocate = AutoCATE()
    autocate.fit(X_train, t_train, yf_train)
    ite_pred = autocate.predict(X_test)

    pehe = root_mean_squared_error(ite_test, ite_pred)
    mape = mean_absolute_percentage_error(ite_test, ite_pred)
    r2 = r2_score(ite_test, ite_pred)
    auqc = auc_qini(ite_test, ite_pred)

    fig, ax = plt.subplots()
    ax.set_title('ACIC AutoCATE (iter: ' + str(dataset_iter + 1) + '; sqPEHE: ' + str(round(pehe, 2)) + ')')
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
    rf = RandomForestRegressor()
    # rf.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
    # ite_pred = rf.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - rf.predict(
    #     np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))
    rf_tuned = RandomizedSearchCV(estimator=rf, param_distributions=rf_search_space, n_iter=10, cv=2, verbose=2,
                                  random_state=42, n_jobs=-1)
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
    print('RF AUQC:\t:', auqc)

    # Compare with RF S-Learner
    lr = LinearRegression()
    lr.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
    ite_pred = lr.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - lr.predict(
        np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

    pehe = root_mean_squared_error(ite_test, ite_pred)
    mape = mean_absolute_percentage_error(ite_test, ite_pred)
    r2 = r2_score(ite_test, ite_pred)
    auqc = auc_qini(ite_test, ite_pred)

    pehes_lr.append(pehe)
    mapes_lr.append(mape)
    r2s_lr.append(r2)
    auqcs_lr.append(auqc)
    print('LR PEHE:\t', pehe)
    print('LR MAPE:\t', mape)
    print('LR R2:  \t', r2)
    print('LR AUQC:\t', auqc)

print("\n\nRESULTS")
print("AutoCATE")
print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes), 4), " | SE: ", np.round(sem(pehes), 4))
print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes), 4), " | SE: ", np.round(sem(mapes), 4))
print("\tR2 -- Avg:        \t", np.round(np.mean(r2s), 4), " | SE: ", np.round(sem(r2s), 4))
print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs), 4), " | SE: ", np.round(sem(auqcs), 4))
print("RF S-Learner")
print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_rf), 4), " | SE: ", np.round(sem(pehes_rf), 4))
print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_rf), 4), " | SE: ", np.round(sem(mapes_rf), 4))
print("\tR2 -- Avg:        \t", np.round(np.mean(r2s_rf), 4), " | SE: ", np.round(sem(r2s_rf), 4))
print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_rf), 4), " | SE: ", np.round(sem(auqcs_rf), 4))
print("LR S-Learner")
print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_lr), 4), " | SE: ", np.round(sem(pehes_lr), 4))
print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_lr), 4), " | SE: ", np.round(sem(mapes_lr), 4))
print("\tR2 -- Avg:        \t", np.round(np.mean(r2s_lr), 4), " | SE: ", np.round(sem(r2s_lr), 4))
print("\tAUQC -- Avg:      \t", np.round(np.mean(auqcs_lr), 4), " | SE: ", np.round(sem(auqcs_lr), 4))

# Histogram (force groups to use the same bins):
# bins = np.histogram(np.hstack((pehes, pehes_rf, pehes_lr)), bins=4)[1]
# fig, ax = plt.subplots()
# sns.kdeplot(pehes, label='AutoCATE', color="blue", alpha=0.3, clip=[0, None], fill=True, linewidth=0, ax=ax)
# sns.kdeplot(pehes_rf, label='RF S-Learner', color="orange", alpha=0.3, clip=[0, None], fill=True, linewidth=0, ax=ax)
# sns.kdeplot(pehes_lr, label='LR S-Learner', color="lightgreen", alpha=0.3, clip=[0, None], fill=True, linewidth=0, ax=ax)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots()
ax.set_title('ACIC - PEHE')
ax.boxplot([pehes, pehes_rf, pehes_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_title('ACIC - MAPE')
ax.boxplot([mapes, mapes_rf, mapes_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_title('IHDP - R2')
ax.boxplot([r2s, r2s_rf, r2s_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_title('IHDP - AUQC')
ax.boxplot([auqcs, auqcs_rf, auqcs_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

print('\n\nTime elapsed: ', np.round(time.time() - start, 2))
