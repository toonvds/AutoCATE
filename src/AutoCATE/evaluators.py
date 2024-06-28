"""
Model validation/evaluation functions.
Author: Toon Vanderschueren
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error,
                             root_mean_squared_error, d2_absolute_error_score)
from scipy.stats import kendalltau
from src.AutoCATE.utils import auc_qini

import matplotlib.pyplot as plt


CLIP = 1e-6


def _get_metric(metric):
    if metric == "R2":
        return r2_score
    elif metric == "D2":
        return d2_absolute_error_score
    elif metric == "MSE":
        return mean_squared_error
    elif metric == "RMSE":
        return root_mean_squared_error
    elif metric == "MAPE":
        return mean_absolute_percentage_error
    elif metric == "MAE":
        return mean_absolute_error
    elif metric == "AUQC":
        # kendalltau_first_arg = lambda x, y: kendalltau(x, y + np.random.normal(0, 1e-8, size=y.shape))[0]
        # return kendalltau_first_arg
        return auc_qini
    else:
        raise ValueError("Metric must be either 'R2', 'D2', 'MSE', 'RMSE', 'MAPE', 'MAE', or 'ME'.")


class _BaseEvaluator:
    def __init__(self):
        pass

    def get_pseudo_outcome(self, X, t, y):
        pass

    def score(self, cate_pred):
        pass


class DREvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, t_pred, y0_pred, y1_pred, metric="R2"):
        super().__init__()
        self.t_pred = t_pred
        self.y0_pred = y0_pred
        self.y1_pred = y1_pred

        self.metric = _get_metric(metric)

        self.pseudo_outcomes = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        t_pred = np.clip(self.t_pred, CLIP, 1 - CLIP)
        dr_pseudo = (t / t_pred - (1 - t) / (1 - t_pred)) * y \
                    + (1 - t / t_pred) * self.y1_pred \
                    - (1 - (1 - t) / (1 - t_pred)) * self.y0_pred

        return dr_pseudo

        # dr_pseudos = []
        # for t_pred, outcome0_model, outcome1_model in (
        #         zip(self.propensity_estimates, self.outcome0_models, self.outcome1_models)):
        #     # Compute DR pseudo-outcome
        #     t_pred = np.clip(t_pred[val_indices], CLIP, 1 - CLIP)
        #
        #     y0_pred = outcome0_model.predict(X)
        #     y1_pred = outcome1_model.predict(X)
        #
        #     pseudo_outcome = (t / t_pred - (1 - t) / (1 - t_pred)) * y \
        #                       + (1 - t / t_pred) * y1_pred \
        #                       - (1 - (1 - t) / (1 - t_pred)) * y0_pred
        #
        #     dr_pseudos.append(pseudo_outcome)
        #
        # return np.mean(dr_pseudos, axis=0)

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred)


class REvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, t_pred, mu_pred, metric="R2"):
        super().__init__()

        self.t_pred = t_pred
        self.mu_pred = mu_pred

        self.metric = _get_metric(metric)

        self.pseudo_outcomes, self.weights = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):

        t_pred = np.clip(self.t_pred, CLIP, 1 - CLIP)        #
        y_tilde = y - self.mu_pred
        t_tilde = t - t_pred

        r_pseudos = y_tilde / t_tilde
        r_weights = t_tilde**2

        return r_pseudos, r_weights

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred, sample_weight=self.weights)
        # r_scores = []
        # for mu_pred, t_pred in zip(self.outcome_estimates, self.propensity_estimates):
        #     # Create R pseudo-outcome:
        #     # if hasattr(propensity_model, "predict_proba"):
        #     #     t_pred = propensity_model.predict_proba(X)[:, 1]
        #     # else:
        #     #     t_pred = propensity_model.predict(X)
        #     t_pred = np.clip(t_pred[val_indices], CLIP, 1 - CLIP)
        #
        #     # mu_pred = outcome_model.predict(X)
        #     mu_pred = mu_pred[val_indices]
        #
        #     y_tilde = y - mu_pred
        #     t_tilde = t - t_pred
        #
        #     pseudo_outcome = y_tilde / t_tilde
        #     weights = t_tilde**2
        #
        #     # Return weighted MSE
        #     # r_scores.append(np.mean((pseudo_outcome - cate_pred)**2 * weights))
        #     # r_scores.append(r2_score(pseudo_outcome, cate_pred, sample_weight=weights))
        #     # r_scores.append(mean_absolute_percentage_error(pseudo_outcome, cate_pred, sample_weight=weights))
        #     r_scores.append(self.metric(pseudo_outcome, cate_pred, sample_weight=weights))
        #
        # # plt.plot(pseudo_outcome, cate_pred, linestyle='None', marker='o')
        # # plt.title(r2_score(pseudo_outcome, cate_pred))
        # # plt.show()
        #
        # return np.mean(r_scores)


class ZEvaluator(_BaseEvaluator):
    # Evaluator based on the Outcome Transformation
    # Also called singly-robust propensity-weighted estimator
    def __init__(self, X, t, y, t_pred, metric="R2"):
        super().__init__()

        self.t_pred = t_pred
        self.metric = _get_metric(metric)

        # Get pseudo outcomes
        self.pseudo_outcomes = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        t_pred = np.clip(self.t_pred, CLIP, 1 - CLIP)

        z_pseudo = np.zeros_like(y)
        z_pseudo[t == 0] = y[t == 0] / t_pred[t == 0]
        z_pseudo[t == 1] = -y[t == 1] / (1 - t_pred[t == 1])

        # for t_pred in self.propensity_estimates:
        #     # Create Z pseudo-outcome:
        #     # if hasattr(propensity_model, "predict_proba"):
        #     #     t_pred = propensity_model.predict_proba(X)[:, 1]
        #     # else:
        #     #     t_pred = propensity_model.predict(X)
        #     t_pred = np.clip(t_pred[val_indices], CLIP, 1 - CLIP)
        #
        #     z_transform = np.zeros_like(y)
        #     z_transform[t == 0] = y[t == 0] / t_pred[t == 0]
        #     z_transform[t == 1] = -y[t == 1] / (1 - t_pred[t == 1])
        #
        #     z_pseudos.append(z_transform)

        return z_pseudo

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred)


class NNEvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, k=1, metric="R2"):
        #   We fit a nearest neighbour model to estimate the counterfactual outcomes
        super().__init__()

        nn0 = KNeighborsRegressor(n_neighbors=k, weights="distance")
        nn1 = KNeighborsRegressor(n_neighbors=k, weights="distance")

        nn0 = nn0.fit(X[t == 0], y[t == 0])
        nn1 = nn1.fit(X[t == 1], y[t == 1])

        self.nn0 = nn0
        self.nn1 = nn1

        self.metric = _get_metric(metric)

        self.pseudo_outcomes = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        y_cf = np.zeros_like(y)
        y_cf[t == 0] = self.nn1.predict(X[t == 0])
        y_cf[t == 1] = self.nn0.predict(X[t == 1])

        ite_nn = np.zeros_like(y)
        ite_nn[t == 0] = y_cf[t == 0] - y[t == 0]
        ite_nn[t == 1] = y[t == 1] - y_cf[t == 1]

        return ite_nn

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred)
        # y_cf = np.zeros_like(y)
        # y_cf[t == 0] = self.nn1.predict(X[t == 0])
        # y_cf[t == 1] = self.nn0.predict(X[t == 1])
        #
        # ite_nn = np.zeros_like(y)
        # ite_nn[t == 0] = y_cf[t == 0] - y[t == 0]
        # ite_nn[t == 1] = y[t == 1] - y_cf[t == 1]

        # plt.plot(ite_nn, cate_pred, linestyle='None', marker='o')
        # plt.title(r2_score(ite_nn, cate_pred))
        # plt.show()

        # return np.mean((ite_nn - cate_pred) ** 2)
        # return np.mean(np.abs(ite_nn - cate_pred))
        # Return correlation coefficient between ite_nn and cate_pred
        # return -np.corrcoef(ite_nn, cate_pred[:, 0] + np.random.normal(0, 1e-8, size=ite_nn.shape))[0, 1]
        # return r2_score(ite_nn, cate_pred)
        # return self.metric(ite_nn, cate_pred)
        # return mean_absolute_percentage_error(ite_nn, cate_pred)


class UEvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, t_pred, mu_pred, metric="R2"):
        super().__init__()

        self.t_pred = t_pred
        self.mu_pred = mu_pred

        self.metric = _get_metric(metric)

        self.pseudo_outcomes = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        t_pred = np.clip(self.t_pred, CLIP, 1 - CLIP)

        u_pseudo = (y - self.mu_pred) / (t - t_pred)

        return u_pseudo

        # u_scores = []
        # for mu_pred, t_pred in zip(self.outcome_estimates, self.propensity_estimates):
        #     t_pred = np.clip(t_pred[val_indices], CLIP, 1 - CLIP)
        #     mu_pred = mu_pred[val_indices]
        #
        #     pseudo_outcome = (y - mu_pred) / (t - t_pred)
        #
        #     u_scores.append(pseudo_outcome)
        #
        # return np.mean(u_scores, axis=0)

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred)
        # u_pseudos = []
        # for mu_pred, t_pred in zip(self.outcome_estimates, self.propensity_estimates):
        #     t_pred = np.clip(t_pred[val_indices], CLIP, 1 - CLIP)
        #     mu_pred = mu_pred[val_indices]
        #
        #     pseudo_outcome = (y - mu_pred) / (t - t_pred)
        #
        #     # Return weighted metric
        #     # u_scores.append(np.mean((pseudo_outcome - cate_pred)**2))
        #     # u_pseudos.append(r2_score(pseudo_outcome, cate_pred))
        #     # u_scores.append(mean_absolute_percentage_error(pseudo_outcome, cate_pred))
        #     u_pseudos.append(self.metric(pseudo_outcome, cate_pred))
        #
        # return np.mean(u_pseudos)


class FEvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, t_pred, metric="R2"):
        super().__init__()

        self.t_pred = t_pred
        self.metric = _get_metric(metric)

        self.pseudo_outcomes = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        t_pred = np.clip(self.t_pred, CLIP, 1 - CLIP)
        f_pseudo = y * (t - t_pred) / (t_pred * (1 - t_pred))

        return f_pseudo
        # f_pseudos = []
        # for t_pred in self.propensity_estimates:
        #     t_pred = np.clip(t_pred[val_indices], CLIP, 1 - CLIP)
        #
        #     f_transform = y * (t - t_pred) / (t_pred * (1 - t_pred))
        #
        #     f_pseudos.append(f_transform)
        #
        # return np.mean(f_pseudos, axis=0)

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred)


class TEvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, y0_pred, y1_pred, metric="R2"):
        super().__init__()

        self.y0_pred = y0_pred
        self.y1_pred = y1_pred
        self.metric = _get_metric(metric)

        self.pseudo_outcomes = self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        pseudo_outcome = np.zeros_like(y)
        pseudo_outcome[t == 0] = self.y1_pred[t == 0] - y[t == 0]
        pseudo_outcome[t == 1] = y[t == 1] - self.y0_pred[t == 1]

        return pseudo_outcome

    def score(self, cate_pred):
        return self.metric(self.pseudo_outcomes, cate_pred)


class IFEvaluator(_BaseEvaluator):
    def __init__(self, X, t, y, t_pred, y0_pred, y1_pred, metric=None):
        super().__init__()

        self.t_pred = t_pred
        self.y0_pred = y0_pred
        self.y1_pred = y1_pred

        self.y_factual = y

        self.get_pseudo_outcome(X, t, y)

    def get_pseudo_outcome(self, X, t, y):
        # Get the required element for computing the influence function criterion
        # We follow the formulation in Curth and van der Schaar (ICML, 2024)
        self.t_pred = np.clip(self.t_pred, CLIP, 1 - CLIP)

        self.pseudo_outcomes = self.y1_pred - self.y0_pred

        self.d_term = t - self.t_pred
        self.c_term = self.t_pred * (1 - self.t_pred)
        self.b_term = 2 * t * (t - self.t_pred) / self.c_term

    def score(self, cate_pred):
        objective_value = ((1 - self.b_term) * (self.pseudo_outcomes ** 2) +
                           self.b_term * self.y_factual * (self.pseudo_outcomes - cate_pred) -
                           self.d_term * (self.pseudo_outcomes - cate_pred) ** 2 + cate_pred ** 2)

        return np.mean(objective_value)
