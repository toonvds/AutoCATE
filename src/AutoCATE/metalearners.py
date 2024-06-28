import numpy as np

from src.AutoCATE.baselearners import get_base_learner
from causalml.inference.meta import (BaseSRegressor, BaseSClassifier, BaseTRegressor, BaseTClassifier, BaseXRegressor,
                                     BaseXClassifier, BaseRRegressor, BaseRClassifier, BaseDRRegressor, BaseDRLearner)
from causalml.inference.meta.base import BaseLearner
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone


def metalearner_collector(metalearner):
    get_metalearner_functions = {
        "S": get_s_learner,
        "T": get_t_learner,
        "X": get_x_learner,
        "DR": get_dr_learner,
        "R": get_r_learner,
        "Z": get_z_learner,
        "Lo": get_lo_learner,
        "U": get_u_learner,
        "F": get_f_learner,
        "RA": get_ra_learner,
    }

    if metalearner not in get_metalearner_functions.keys():
        raise ValueError("Metalearner must be one of 'S', 'T', 'X', 'DR', 'R', 'Z', 'Lo', 'U', 'F', or 'RA'.")

    return get_metalearner_functions[metalearner]


def get_s_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    # Choose base learner
    base_learner_s = trial.suggest_categorical("base_learner_s", base_learners)

    # Get base learner and associated hyperparameters
    base_model = get_base_learner(trial, task=task, base_learner=base_learner_s, meta_learner="S",
                                  joint_optimization=joint_optimization, n_jobs=n_jobs)

    if task == "regression":
        return BaseSRegressor(learner=base_model)
    elif task == "classification":
        return BaseSClassifier(learner=base_model)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_t_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):

    # Get base learner and associated hyperparameters
    if single_base_learner:
        # Choose base learners
        base_learner_t = trial.suggest_categorical("base_learner_t", base_learners)

        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_t, meta_learner="T",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = clone(base_model_control)
    else:
        # Choose base learners
        base_learner_t0 = trial.suggest_categorical("base_learner_t0", base_learners)
        base_learner_t1 = trial.suggest_categorical("base_learner_t1", base_learners)

        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_t0, meta_learner="T0",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = get_base_learner(trial, task=task, base_learner=base_learner_t1, meta_learner="T1",
                                                joint_optimization=joint_optimization, n_jobs=n_jobs)

    if task == "regression":
        return BaseTRegressor(control_learner=base_model_control, treatment_learner=base_model_treatment)
    elif task == "classification":
        return BaseTClassifier(control_learner=base_model_control, treatment_learner=base_model_treatment)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_x_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):

    # Get base learner and associated hyperparameters
    if single_base_learner:
        # Choose base learner
        base_learner_x = trial.suggest_categorical("base_learner_x", base_learners)

        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_x, meta_learner="X",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = clone(base_model_control)
        cate_model_control = clone(base_model_control)
        cate_model_treatment = clone(base_model_control)
    else:
        # Choose base learner
        base_learner_x_t0 = trial.suggest_categorical("base_learner_x_t0", base_learners)
        base_learner_x_t1 = trial.suggest_categorical("base_learner_x_t1", base_learners)
        base_learner_x0 = trial.suggest_categorical("base_learner_x0", base_learners)
        base_learner_x1 = trial.suggest_categorical("base_learner_x1", base_learners)

        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_x_t0, meta_learner="Xt0",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = get_base_learner(trial, task=task, base_learner=base_learner_x_t1, meta_learner="Xt1",
                                                joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model_control = get_base_learner(trial, task="regression", base_learner=base_learner_x0, meta_learner="X0",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model_treatment = get_base_learner(trial, task="regression", base_learner=base_learner_x1, meta_learner="X1",
                                                joint_optimization=joint_optimization, n_jobs=n_jobs)

    if task == "regression":
        return BaseXRegressor(control_outcome_learner=base_model_control,
                              treatment_outcome_learner=base_model_treatment,
                              control_effect_learner=cate_model_control,
                              treatment_effect_learner=cate_model_treatment)
    elif task == "classification":
        return BaseXClassifier(control_outcome_learner=base_model_control,
                               treatment_outcome_learner=base_model_treatment,
                               control_effect_learner=cate_model_control,
                               treatment_effect_learner=cate_model_treatment)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_dr_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):

    # Get base learner and associated hyperparameters
    if single_base_learner:
        # Choose base learners
        base_learner_dr = trial.suggest_categorical("base_learner_dr", base_learners)

        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_dr, meta_learner="DR",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = clone(base_model_control)
        cate_model = clone(base_model_control)
    else:
        # Choose base learners
        base_learner_dr_y0 = trial.suggest_categorical("base_learner_dr_y0", base_learners)
        base_learner_dr_y1 = trial.suggest_categorical("base_learner_dr_y1", base_learners)
        base_learner_dr_tau = trial.suggest_categorical("base_learner_dr_tau", base_learners)

        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_dr_y0, meta_learner="DRy0",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = get_base_learner(trial, task=task, base_learner=base_learner_dr_y1, meta_learner="DRy1",
                                                joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_dr_tau, meta_learner="DRtau",
                                      joint_optimization=joint_optimization, n_jobs=n_jobs)

    if task == "regression":
        return BaseDRRegressor(control_outcome_learner=base_model_control,
                               treatment_outcome_learner=base_model_treatment,
                               treatment_effect_learner=cate_model)
    elif task == "classification":
        return BaseDRLearner(control_outcome_learner=base_model_control,
                             treatment_outcome_learner=base_model_treatment,
                             treatment_effect_learner=cate_model)
        return
    else:
        raise ValueError("Task must be 'regression' for DR learner")


def get_r_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    if single_base_learner:
        # Choose base learners for y and t
        # For R-Learner, we exclude the NN and GP as it does not support a "sample_weight" argument in fit()
        base_learners_weighted = [learner for learner in base_learners if not(learner in ["NN", "GP", "kNN"])]
        base_learner_r = trial.suggest_categorical("base_learner_r", base_learners_weighted)

        # Get base learner and associated hyperparameters
        base_model_outcome = get_base_learner(trial, task=task, base_learner=base_learner_r, meta_learner="R",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_propensity = clone(base_model_outcome)
        cate_model = clone(base_model_outcome)

    else:
        # Choose base learners for y and t
        base_learner_r_y = trial.suggest_categorical("base_learner_r_y", base_learners)
        base_learner_r_t = trial.suggest_categorical("base_learner_r_t", base_learners)

        # For R-Learner, we exclude the NN and GP as it does not support a "sample_weight" argument in fit()
        base_learners_weighted = [learner for learner in base_learners if not(learner in ["NN", "GP", "kNN"])]
        base_learner_r_tau = trial.suggest_categorical("base_learner_r_tau", base_learners_weighted)

        # Get base learner and associated hyperparameters
        base_model_outcome = get_base_learner(trial, task=task, base_learner=base_learner_r_y, meta_learner="Ry",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_propensity = get_base_learner(trial, task=task, base_learner=base_learner_r_t, meta_learner="Rt",
                                                 joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_r_tau, meta_learner="Rtau",
                                      joint_optimization=joint_optimization, n_jobs=n_jobs)

    if task == "regression":
        return BaseRRegressor(outcome_learner=base_model_outcome,
                              propensity_learner=base_model_propensity,
                              effect_learner=cate_model)
    elif task == "classification":
        return BaseRClassifier(outcome_learner=base_model_outcome,
                               propensity_learner=base_model_propensity,
                               effect_learner=cate_model)
    else:
        raise ValueError("Task must be either 'regression' or 'classification'")


def get_z_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    # Choose base learners for y and t
    # base_learner_z_t = trial.suggest_categorical("base_learner_z_y", base_learners)
    base_learner_z_tau = trial.suggest_categorical("base_learner_z_t", base_learners)

    # Get base learner and associated hyperparameters
    # base_model_propensity = get_base_learner(trial, task="classification", base_learner=base_learner_z_t,
    #                                          meta_learner="Zt")
    cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_z_tau, meta_learner="Ztau",
                                  joint_optimization=joint_optimization, n_jobs=n_jobs)

    return ZLearner(cate_model=cate_model)


def get_lo_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    base_learner_lo = trial.suggest_categorical("base_learner_lo", base_learners)

    # Get base learner and associated hyperparameters
    cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_lo, meta_learner="Ztau",
                                  joint_optimization=joint_optimization, n_jobs=n_jobs)

    return LoLearner(cate_model=cate_model)


def get_u_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    if single_base_learner:
        # Choose base learners for y and tau
        base_learner_u = trial.suggest_categorical("base_learner_u", base_learners)

        # Get base learner and associated hyperparameters
        base_model_mu = get_base_learner(trial, task=task, base_learner=base_learner_u, meta_learner="U",
                                         joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model = clone(base_model_mu)

    else:
        # Choose base learners for y and tau
        base_learner_u_mu = trial.suggest_categorical("base_learner_u_mu", base_learners)
        base_learner_u_tau = trial.suggest_categorical("base_learner_u_tau", base_learners)

        # Get base learner and associated hyperparameters
        base_model_mu = get_base_learner(trial, task=task, base_learner=base_learner_u_mu, meta_learner="Umu",
                                         joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_u_tau, meta_learner="Utau",
                                      joint_optimization=joint_optimization, n_jobs=n_jobs)

    return ULearner(mu_model=base_model_mu, cate_model=cate_model)


def get_f_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    # Choose base learners for y
    base_learner_f = trial.suggest_categorical("base_learner_f", base_learners)

    # Get base learner and associated hyperparameters
    cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_f, meta_learner="Ftau",
                                  joint_optimization=joint_optimization, n_jobs=n_jobs)

    return FLearner(cate_model=cate_model)


def get_ra_learner(trial, base_learners, task="regression", joint_optimization=False, single_base_learner=False, n_jobs=None):
    if single_base_learner:
        # Choose base learner
        base_learner_ra = trial.suggest_categorical("base_learner_ra", base_learners)

        # Get base learner and associated hyperparameters
        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_ra, meta_learner="RA",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = clone(base_model_control)
        cate_model = clone(base_model_control)

    else:
        # Choose base learners for y
        base_learner_ra_y = trial.suggest_categorical("base_learner_ra_y", base_learners)
        base_learner_ra_t = trial.suggest_categorical("base_learner_ra_t", base_learners)
        base_learner_ra_tau = trial.suggest_categorical("base_learner_ra_tau", base_learners)

        # Get base learner and associated hyperparameters
        base_model_control = get_base_learner(trial, task=task, base_learner=base_learner_ra_y, meta_learner="RAy",
                                              joint_optimization=joint_optimization, n_jobs=n_jobs)
        base_model_treatment = get_base_learner(trial, task=task, base_learner=base_learner_ra_t, meta_learner="RAt",
                                                joint_optimization=joint_optimization, n_jobs=n_jobs)
        cate_model = get_base_learner(trial, task="regression", base_learner=base_learner_ra_tau, meta_learner="RAtau",
                                      joint_optimization=joint_optimization, n_jobs=n_jobs)

    return RALearner(control_outcome_learner=base_model_control, treatment_outcome_learner=base_model_treatment,
                     cate_model=cate_model)


class ZLearner:
    # Singly Robust Learner based on the Z-Transform
    def __init__(self, cate_model):
        # self.propensity_learner = propensity_learner
        self.cate_model = cate_model

    def __repr__(self):
        return "{}(model={})".format(self.__class__.__name__, self.cate_model.__repr__())

    def fit(self, X, t, y, p):
        # Train propensity score model and get out-of-fold propensity estimates
        # t_pred = cross_val_predict(self.propensity_learner, X, t, cv=3, method='predict_proba')[:, 1]
        # t_pred = np.clip(t_pred, self.clip_value, 1 - self.clip_value)

        # Create pseudo-label
        z_transform = np.zeros_like(y)
        z_transform[t == 0] = y[t == 0] / p[t == 0]
        z_transform[t == 1] = -y[t == 1] / (1 - p[t == 1])

        # Train outcome model based on Z-Transform
        self.cate_model.fit(X, z_transform)

    def predict(self, X):
        return self.cate_model.predict(X)[:, np.newaxis]


class LoLearner:
    # Lo's approach, including interaction terms between the treatment and the features
    def __init__(self, cate_model):
        self.cate_model = cate_model

    def __repr__(self):
        return "{}(model={})".format(self.__class__.__name__, self.cate_model.__repr__())

    def fit(self, X, t, y):
        # Add (first order) interaction terms between the treatment and the features
        Xt_interact = X * t[:, np.newaxis]
        # Xt_interact_2 = Xt_interact * t[:, np.newaxis]
        Xt_extended = np.concatenate((X, Xt_interact, t[:, np.newaxis]), axis=1)

        # Train outcome model based on extended features
        self.cate_model.fit(Xt_extended, y)

    def predict(self, X):
        # Prediction for treatment group
        Xt_interact1 = X * np.ones((len(X), 1))
#         Xt_interact1_2 = Xt_interact1 * np.ones((len(X), 1))
        Xt_extended1 = np.concatenate((X, Xt_interact1, np.ones((len(X), 1))), axis=1)

        y1_pred = self.cate_model.predict(Xt_extended1)[:, np.newaxis]

        # Prediction for control group
        Xt_interact0 = X * np.zeros((len(X), 1))
#         Xt_interact0_2 = Xt_interact0 * np.zeros((len(X), 1))
        Xt_extended0 = np.concatenate((X, Xt_interact0, np.zeros((len(X), 1))), axis=1)

        y0_pred = self.cate_model.predict(Xt_extended0)[:, np.newaxis]

        return y1_pred - y0_pred


class ULearner:
    # U-Learner mentioned in page 6 of Nie, Xinkun, and Stefan Wager. "Quasi-oracle estimation of heterogeneous
    # treatment effects." Biometrika 108, no. 2 (2021): 299-319.
    # See also Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous
    # treatment effects using machine learning. Proceedings of the national academy of sciences, 116(10), 4156-4165.
    def __init__(self, mu_model, cate_model):
        self.mu_model = mu_model
        self.cate_model = cate_model

    def __repr__(self):
        return "{}(mu_model={}; cate_model={})".format(self.__class__.__name__, self.mu_model.__repr__(),
                                                       self.cate_model.__repr__())

    def fit(self, X, t, y, p):
        # Train expected value model
        mu_est = cross_val_predict(self.mu_model, X, y, cv=3)

        # Create pseudo-label
        u_transform = (y - mu_est) / (t - p)

        # Train outcome model based on Z-Transform
        self.cate_model.fit(X, u_transform)

    def predict(self, X):
        return self.cate_model.predict(X)[:, np.newaxis]


class FLearner:
    # See Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous
    # treatment effects using machine learning. Proceedings of the national academy of sciences, 116(10), 4156-4165.
    def __init__(self, cate_model):
        self.cate_model = cate_model

    def __repr__(self):
        return "{}(model={})".format(self.__class__.__name__, self.cate_model.__repr__())

    def fit(self, X, t, y, p):
        # Create pseudo-label
        f_transform = y * (t - p) / (p * (1 - p))

        # Train outcome model based on Z-Transform
        self.cate_model.fit(X, f_transform)

    def predict(self, X):
        return self.cate_model.predict(X)[:, np.newaxis]


class RALearner:
    def __init__(self, control_outcome_learner, treatment_outcome_learner, cate_model):
        self.control_outcome_learner = control_outcome_learner
        self.treatment_outcome_learner = treatment_outcome_learner
        self.cate_model = cate_model

    def __repr__(self):
        return "{}(control_model={}; treatment_model={}; cate_model={})".format(
            self.__class__.__name__,
            self.control_outcome_learner.__repr__(),
            self.treatment_outcome_learner.__repr__(),
            self.cate_model.__repr__()
        )

    def fit(self, X, t, y):
        # Train outcome model based on Z-Transform
        self.control_outcome_learner.fit(X[t == 0], y[t == 0])
        self.treatment_outcome_learner.fit(X[t == 1], y[t == 1])

        ite = np.zeros_like(y)
        ite[t == 0] = self.treatment_outcome_learner.predict(X[t == 0]) - y[t==0]
        ite[t == 1] = y[t == 1] - self.control_outcome_learner.predict(X[t == 1])

        # Train outcome model based on Z-Transform
        self.cate_model.fit(X, ite)

    def predict(self, X):
        return self.cate_model.predict(X)[:, np.newaxis]

