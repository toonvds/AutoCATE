"""
This script is used to test and benchmark the AutoCATE algorithm on the Twins data set.
Author: Toon Vanderschueren
"""

# import os
# os.chdir("..")  # Change working directory from 'experiments' to 'AutoCATE' to load data
# print(os.getcwd())

import time
from data.utils import load_twins

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

from src.AutoCATE.AutoCATE import AutoCATE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


np.random.seed(42)

pehes = []
pehes_rf = []
pehes_lr = []

start = time.time()

# Iterate over 10 seeds for the Twins dataset:
# for dataset_iter in range(10):
for dataset_iter in range(1):  # For testing purposes
    print('Iteration: ', dataset_iter + 1)
    X, t, yf, ite = load_twins()

    X_train, X_test, t_train, _, yf_train, _, _, ite_test = train_test_split(X, t, yf, ite, test_size=0.3, random_state=42)

    # AutoCATE
    autocate = AutoCATE(task="classification")
    autocate.fit(X_train, t_train, yf_train)
    ite_pred = autocate.predict(X_test)

    pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))

    fig, ax = plt.subplots()
    ax.set_title('Twins AutoCATE (iter: ' + str(dataset_iter + 1) + '; sqPEHE: ' + str(round(pehe, 2)) + ')')
    ax.plot(ite_test, ite_pred, 'o', linestyle='None', color='green')
    # Add a linear trend line
    ax.plot(np.unique(ite_test), np.poly1d(np.polyfit(ite_test, ite_pred, 1))(np.unique(ite_test)),
            alpha=0.1, color='green')
    # Add a x=y line
    # ax.plot(np.unique(ite_test), np.unique(ite_test), 'b', linestyle='--', alpha=0.2)
    plt.show()

    pehes.append(pehe)
    print('AutoCATE PEHE:', pehe)

    # Compare with RF S-Learner
    rf = RandomForestClassifier()
    rf.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
    ite_pred = rf.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - rf.predict(
        np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

    pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))

    pehes_rf.append(pehe)
    print('RF PEHE:', pehe)

    # Compare with RF S-Learner
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
    ite_pred = lr.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - lr.predict(
        np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

    pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))

    pehes_lr.append(pehe)
    print('LR PEHE:', pehe)

print("\n\nRESULTS")
print("AutoCATE")
print("\tsqrt PEHE -- Avg: ", np.round(np.mean(pehes), 4), " | SE: ", np.round(sem(pehes), 4))
print("RF S-Learner")
print("\tsqrt PEHE -- Avg: ", np.round(np.mean(pehes_rf), 4), " | SE: ", np.round(sem(pehes_rf), 4))
print("LR S-Learner")
print("\tsqrt PEHE -- Avg: ", np.round(np.mean(pehes_lr), 4), " | SE: ", np.round(sem(pehes_lr), 4))

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
ax.set_title('Twins - PEHE')
ax.boxplot([pehes, pehes_rf, pehes_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

print('\n\nTime elapsed: ', np.round(time.time() - start, 2))
