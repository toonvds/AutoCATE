import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, BaseCrossValidator


class ContStratifiedKFold(BaseCrossValidator):
    # TODO: Implement StratifiedCV
    """
    Custom Stratified Cross-Validation class that accounts for continuous outcomes.
    """
    def __init__(self, n_splits=5, n_bins=10, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle  # Todo: Implement shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        # Create a new stratification variable that discretizes y
        y_discretized = KBinsDiscretizer(n_bins=self.n_bins,
                                          encode='ordinal',
                                          strategy='quantile').fit_transform(y.copy().reshape(-1, 1))

        # Sort indices based on y_discretized values
        sorted_indices = np.argsort(y_discretized, axis=0).ravel()

        # Get self.n_splits mutually exclusive test splits, stratified based on y_discretized:
        for i in range(self.n_splits):
            test_indices = sorted_indices[i::self.n_splits]

            train_indices = np.setdiff1d(sorted_indices, test_indices)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class ContStratifiedKFoldWithTreatment(BaseCrossValidator):
    """
    Custom Stratified Cross-Validation class that accounts for continuous outcomes.
    """
    def __init__(self, n_splits=5, n_bins=10, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle  # Todo: Implement shuffle
        self.random_state = random_state

    def split(self, X, t=None, y=None, groups=None):
        if t is None:
            t = np.ones_like(y)

        # Create a new stratification variable that combines y_discretized and t
        yt = y.copy() * t.copy() + y.copy() * (t.copy() - 1)
        yt_discretized = KBinsDiscretizer(n_bins=self.n_bins,
                                          encode='ordinal',
                                          strategy='quantile').fit_transform(yt.reshape(-1, 1))

        # Sort indices based on y_discretized values
        sorted_indices = np.argsort(yt_discretized, axis=0).ravel()

        # Get self.n_splits mutually exclusive test splits, stratified based on y_discretized:
        for i in range(self.n_splits):
            test_indices = sorted_indices[i::self.n_splits]

            train_indices = np.setdiff1d(sorted_indices, test_indices)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class SingleSplit(BaseCrossValidator):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        train_indices, test_indices = train_test_split(np.arange(X.shape[0]),
                                                       test_size=self.test_size,
                                                       random_state=self.random_state)
        yield train_indices, test_indices


class SingleStratifiedSplitWithTreatment(BaseCrossValidator):
    def __init__(self, test_size=0.2, n_bins=10, random_state=42):
        self.test_size = test_size
        self.n_bins = n_bins
        self.random_state = random_state

    def get_n_splits(self, X=None, t=None, y=None, groups=None):
        return 1

    def split(self, X, t=None, y=None, groups=None):
        yt = y.copy() * t.copy() + y.copy() * (t.copy() - 1)
        yt_discretized = KBinsDiscretizer(n_bins=self.n_bins,
                                          encode='ordinal',
                                          strategy='quantile').fit_transform(yt.reshape(-1, 1))
        train_indices, test_indices = train_test_split(np.arange(X.shape[0]),
                                                       test_size=self.test_size,
                                                       random_state=self.random_state,
                                                       stratify=yt_discretized)
        yield train_indices, test_indices


# Linear model with weights constrained to [0, 1]
class ConstrainedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, reg_alpha=0):
        self.reg_alpha = reg_alpha
        self.coef_ = None

    def fit(self, X, y, sample_weight=None):
        # Define the objective function that scipy's minimize will optimize
        def objective(weights, X, y):
            predictions = np.dot(X, weights)
            error = np.mean(((y - predictions) ** 2) * sample_weight + self.reg_alpha * np.sum(weights ** 2))
            return error

        # Initial guess for weights
        initial_weights = np.ones(X.shape[1]) / X.shape[1]

        # Define the constraints: weights must be between 0 and 1, and sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(X.shape[1])]

        # Use scipy's minimize function to find the optimal weights
        result = minimize(objective, initial_weights, args=(X, y), method='SLSQP', bounds=bounds,
                          constraints=constraints, options={"disp": False})

        # Store the optimal weights
        self.coef_ = result.x

        return self

    def predict(self, X):
        return np.dot(X, self.coef_)


def auc_qini(cate_true, cate_pred, sample_weight=None):
    # We add a small value to the predictions to deal with ties
    cate_pred_perturbed = cate_pred.copy() + np.random.normal(0, 1e-8, cate_pred.shape)

    # Get model's AUC Qini, as well as perfect and random
    if sample_weight is None:
        sample_weight = np.ones_like(cate_true)
    auc_perfect = np.trapz(y=np.cumsum(cate_true[np.argsort(cate_true)[::-1]]),
                           x=np.cumsum(sample_weight[np.argsort(cate_true)[::-1]]))
    auc_model = np.trapz(y=np.cumsum(cate_true[np.argsort(cate_pred_perturbed)[::-1]]),
                         x=np.cumsum(sample_weight[np.argsort(cate_pred_perturbed)[::-1]]))
    auc_random = np.trapz(x=[0, len(cate_true)], y=[0, np.sum(cate_true)])

    # Normalize AUC Qini
    auc_normalized = (auc_model - auc_random) / (auc_perfect - auc_random)

    return auc_normalized
