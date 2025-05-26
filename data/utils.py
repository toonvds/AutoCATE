import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


# Load IHDP data
def load_ihdp_iteration(iteration=0):
    data_train = np.load('data/IHDP/ihdp_npci_1-100.train.npz')
    data_test = np.load('data/IHDP/ihdp_npci_1-100.test.npz')

    # Train data
    X_train = data_train['x'][:, :, iteration]
    # X_train[:, 13] = X_train[:, 13] - 1  # Correct this dummy
    t_train = data_train['t'][:, iteration].ravel()

    yf_train_val = data_train['yf'][:, iteration].ravel()

    # Test data
    X_test = data_test['x'][:, :, iteration]
    # X_test[:, 13] = X_test[:, 13] - 1  # Correct this dummy

    mu0_test = data_test['mu0'][:, iteration].ravel()
    mu1_test = data_test['mu1'][:, iteration].ravel()

    ite_test = mu1_test - mu0_test

    return X_train, t_train, yf_train_val, X_test, mu0_test, mu1_test, ite_test


def load_twins():
    full_df = pd.read_csv("data/Twins/twins.csv", index_col=0)

    X = full_df.drop(['T', 'y0', 'y1', 'yf', 'y_cf', 'Propensity'], axis='columns').to_numpy()
    t = full_df['T'].to_numpy()
    yf = full_df['yf'].to_numpy()

    ite = (full_df['y1'] - full_df['y0']).to_numpy()

    return X, t, yf, ite


def load_acic_iteration(iteration=0):
    x_df = pd.read_csv("data/ACIC/data_cf_all/x.csv")
    x_df = pd.get_dummies(x_df)
    X = x_df.to_numpy()

    # Get first file in this directory
    outcome_dir = "data/ACIC/data_cf_all/" + str(iteration + 1) + "/"
    for _, _, files in os.walk(outcome_dir):
        file_name = files[0]
    outcome_df = pd.read_csv(outcome_dir + file_name)

    # Get outcomes
    t = outcome_df['z'].to_numpy()

    y0 = outcome_df['y0'].to_numpy()
    y1 = outcome_df['y1'].to_numpy()
    yf = t*y1 + (1-t)*y0

    mu0 = outcome_df['mu0'].to_numpy()
    mu1 = outcome_df['mu1'].to_numpy()
    ite = mu1 - mu0

    return X, t, yf, ite


def load_news_iteration(iteration=0):
    # Read data from CSV files
    # Code adapted from:
    # - https://github.com/clinicalml/cfrnet/blob/9daea5d8ba7cb89f413065c5ce7f0136f0e84c9b/cfr/util.py#L67
    # - https://github.com/toonvds/NOFLITE/blob/main/data_loading.py

    fname = 'data/News/csv/topic_doc_mean_n5000_k3477_seed_' + str(iteration + 1) + '.csv'
    data = np.loadtxt(open(fname + '.y', "rb"), delimiter=",")  # t, y_f, y_cf, mu0, mu1

    def load_sparse(fname):
        """ Load sparse data set """
        E = np.loadtxt(open(fname, "rb"), delimiter=",")
        H = E[0, :]
        n = int(H[0])
        d = int(H[1])
        E = E[1:, :]
        S = sparse.coo_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(n, d))
        S = S.todense()

        return S

    X = load_sparse(fname + '.x')
    X = np.asarray(X)

    # Putting data in correct structure
    t = data[:, 0]
    yf = data[:, 1]
    yc = data[:, 2]

    # Convert factual and counterfactual to potential outcomes Y(0) and Y(1)
    y1 = yf * t + yc * (1 - t)
    y0 = yc * t + yf * (1 - t)
    yf = t*y1 + (1-t)*y0

    ite = y1 - y0

    return X, t, yf, ite


def load_hillstrom():
    full_df = pd.read_csv("data/Hillstrom/MineThatData.csv")

    full_df["history_segment"] = full_df["history_segment"].map({
        '1) $0 - $100': 50,
        '2) $100 - $200': 150,
        '3) $200 - $350': 275,
        '4) $350 - $500': 425,
        '5) $500 - $750': 575,
        '6) $750 - $1,000': 825,
        '7) $1,000 +': 1000,
    })

    cat_vars = ["history_segment", "zip_code", "channel"]

    full_df = pd.get_dummies(full_df, columns=cat_vars)

    X = full_df.drop(['segment', 'visit', 'conversion', 'spend'], axis='columns').to_numpy()
    t = full_df["segment"].apply(lambda x: 0 if x == 'No E-Mail' else 1).to_numpy().astype("float")  # 1 if e-mail received
    yf = full_df['visit'].to_numpy().astype("float")

    return X, t, yf


def load_information():
    full_df = pd.read_csv("data/Information/Information.csv")
    full_df = full_df.drop(['UNIQUE_ID'], axis=1)

    X = full_df.drop(['TREATMENT', 'PURCHASE'], axis='columns').to_numpy()
    t = full_df['TREATMENT'].to_numpy().astype("float")
    yf = full_df['PURCHASE'].to_numpy().astype("float")

    return X, t, yf


def load_synthetic(gamma, seed=42, size=1000, dimensions=5):
    np.random.seed(seed)

    # We generate synthetic data
    X = np.random.normal(0, 1, size=(size, dimensions))

    # Generate the potential outcomes
    coef_y0 = np.random.uniform(-1, 1, size=[dimensions])
    coef_y1 = np.random.uniform(-1, 1, size=[dimensions])
    y0 = np.sin(np.matmul(X, coef_y0)) + np.random.normal(0, 0.1, size=[size])
    y1 = y0 + (np.matmul(X, coef_y1)) + np.random.normal(0, 0.1, size=[size]) ** 2

    # Generate the treatment assignment
    coef_t = np.random.uniform(-1, 1, size=[dimensions])
    prob_treat = expit(gamma * np.matmul(X, coef_t))
    t = np.random.binomial(1, prob_treat, size=[size])

    # Generate the observed outcome (yf) and cate/ite
    yf = t * y1 + (1 - t) * y0
    ite = y1 - y0

    return X, t, yf, ite
