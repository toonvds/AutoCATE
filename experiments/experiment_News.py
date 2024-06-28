"""
This script is used to test and benchmark the AutoCATE algorithm on the data from the News data.
Author: Toon Vanderschueren
"""

import time
from data.utils import load_news_iteration

import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

from src.AutoCATE.AutoCATE import AutoCATE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


np.random.seed(42)

pehes = []
pehes_rf = []
pehes_lr = []

mapes = []
mapes_rf = []
mapes_lr = []

start = time.time()

# Iterate over all 50 iterations of the News dataset:
# for dataset_iter in range(50):
for dataset_iter in range(2):  # For testing purposes
    print('Iteration: ', dataset_iter + 1)
    X, t, yf, ite = load_news_iteration(dataset_iter)

    X_train, X_test, t_train, t_test, yf_train, yf_test, _, ite_test = train_test_split(X, t, yf, ite,
                                                                                        # test_size=0.3, random_state=42)
                                                                                        test_size=0.9, random_state=42)

    # AutoCATE
    autocate = AutoCATE()
    autocate.fit(X_train, t_train, yf_train)
    ite_pred = autocate.predict(X_test)

    pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))
    mape = mean_absolute_percentage_error(ite_test, ite_pred)

    fig, ax = plt.subplots()
    ax.set_title('News AutoCATE (iter: ' + str(dataset_iter + 1) + '; sqPEHE: ' + str(round(pehe, 2)) + ')')
    ax.plot(ite_test, ite_pred, 'o', linestyle='None', color='green')
    # Add a linear trend line
    ax.plot(np.unique(ite_test), np.poly1d(np.polyfit(ite_test, ite_pred, 1))(np.unique(ite_test)),
            alpha=0.1, color='green')
    # Add a x=y line
    ax.plot(np.unique(ite_test), np.unique(ite_test), 'b', linestyle='--', alpha=0.2)
    plt.show()

    pehes.append(pehe)
    mapes.append(mape)
    print('AutoCATE PEHE:', pehe)
    print('AutoCATE MAPE:', mape)

    # Compare with RF S-Learner
    rf = RandomForestRegressor()
    rf.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
    ite_pred = rf.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - rf.predict(
        np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

    pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))
    mape = mean_absolute_percentage_error(ite_test, ite_pred)

    pehes_rf.append(pehe)
    mapes_rf.append(mape)
    print('RF PEHE:', pehe)
    print('RF MAPE:', mape)

    # Compare with RF S-Learner
    lr = LinearRegression()
    lr.fit(np.concatenate((X_train, t_train[:, np.newaxis]), axis=1), yf_train)
    ite_pred = lr.predict(np.concatenate((X_test, np.ones((len(X_test), 1))), axis=1)) - lr.predict(
        np.concatenate((X_test, np.zeros((len(X_test), 1))), axis=1))

    pehe = np.sqrt(np.mean((ite_pred - ite_test) ** 2))
    mape = mean_absolute_percentage_error(ite_test, ite_pred)

    pehes_lr.append(pehe)
    mapes_lr.append(mape)
    print('LR PEHE:', pehe)
    print('LR MAPE:', mape)

print("\n\nRESULTS")
print("AutoCATE")
print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes), 4), " | SE: ", np.round(sem(pehes), 4))
print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes), 4), " | SE: ", np.round(sem(mapes), 4))
print("RF S-Learner")
print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_rf), 4), " | SE: ", np.round(sem(pehes_rf), 4))
print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_rf), 4), " | SE: ", np.round(sem(mapes_rf), 4))
print("LR S-Learner")
print("\tsqrt PEHE -- Avg: \t", np.round(np.mean(pehes_lr), 4), " | SE: ", np.round(sem(pehes_lr), 4))
print("\tMAPE -- Avg:      \t", np.round(np.mean(mapes_lr), 4), " | SE: ", np.round(sem(mapes_lr), 4))

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
ax.set_title('News - PEHE')
ax.boxplot([pehes, pehes_rf, pehes_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.set_title('News - MAPE')
ax.boxplot([mapes, mapes_rf, mapes_lr], labels=['AutoCATE', 'RF S-Learner', 'LR S-Learner'])
plt.tight_layout()
plt.show()

print('\n\nTime elapsed: ', np.round(time.time() - start, 2))
