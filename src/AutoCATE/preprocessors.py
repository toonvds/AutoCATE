from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, SplineTransformer, PolynomialFeatures,
                                   PowerTransformer, QuantileTransformer, KBinsDiscretizer)
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, SelectPercentile, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def get_preprocess_pipeline(trial, n_dim=None, preprocessors=None, task="regression"):
    feature_selector_type = trial.suggest_categorical(
        "feature_selector", preprocessors["feature_selector"]
        # ["EmptyStep"]
        # ["EmptyStep", "VarianceThreshold", "SelectKBest", "SelectPercentile"]
        # ["EmptyStep", "VarianceThreshold", "SelectPercentile"]
    )
    feature_selector = get_preprocessor(trial=trial, preprocessor=feature_selector_type, n_dim=n_dim, task=task)

    feature_transformer_type = trial.suggest_categorical(
        "feature_transformer", preprocessors["feature_transformer"]
        # ["EmptyStep"]
        # ["SplineTransformer"]
        # ["EmptyStep", "PolynomialFeatures"]
        # ["EmptyStep", "SplineTransformer"]
        # ["EmptyStep", "KBinsDiscretizer"]
        # ["EmptyStep", "SplineTransformer", "PolynomialFeatures", "KBinsDiscretizer"]
    )
    feature_transformer = get_preprocessor(trial=trial, preprocessor=feature_transformer_type, n_dim=n_dim)

    scaler_type = trial.suggest_categorical(
        "feature_scaler", preprocessors["feature_scaler"]
        # ["EmptyStep"]
        # ["EmptyStep", "StandardScaler", "RobustScaler"]
        # ["EmptyStep", "StandardScaler", "MinMaxScaler", "RobustScaler", "PowerTransformer", "QuantileTransformer"]
    )
    scaler = get_preprocessor(trial=trial, preprocessor=scaler_type, n_dim=n_dim)

    return Pipeline([
        ("feature_selector", feature_selector),
        ("feature_transformer", feature_transformer),
        ("feature_scaler", scaler),
    ])


def get_preprocessor(trial, preprocessor, n_dim, task="regression"):
    # Skip a preprocessing step
    if preprocessor == "EmptyStep":
        return EmptyStep()

    # Feature selection
    elif preprocessor == "VarianceThreshold":
        return VarianceThreshold(threshold=trial.suggest_float("VarThr_threshold", 0, 0.04), )
    elif preprocessor == "SelectKBest":
        return SelectKBest(k=trial.suggest_int("k", 5, n_dim),
                           # score_func=f_regression if task == "regression" else f_classif,
                           score_func=mutual_info_regression if task == "regression" else mutual_info_classif,
                           )
    elif preprocessor == "SelectPercentile":
        return SelectPercentile(percentile=trial.suggest_float("SP_percentile", 0, 100),
                                # score_func=f_regression if task == "regression" else f_classif,
                                score_func=mutual_info_regression if task == "regression" else mutual_info_classif,)
    elif preprocessor == "PCA":
        return PCA(n_components=trial.suggest_int("PCA_n_components", 1, n_dim),)
    elif preprocessor == "KernelPCA":
        return KernelPCA(n_components=trial.suggest_int("KPCA_n_components", 1, n_dim),
                         kernel=trial.suggest_categorical("KPCA_kernel", ["linear", "poly", "rbf", "sigmoid"]))
    elif preprocessor == "FastICA":
        return FastICA(n_components=trial.suggest_int("ICA_n_components", 1, n_dim),
                       algorithm=trial.suggest_categorical("ICA_algorithm", ["parallel", "deflation"]),
                       fun=trial.suggest_categorical("ICA_fun", ["logcosh", "exp", "cube"]),
                       max_iter=1000)

    # Feature transformations
    elif preprocessor == "SplineTransformer":
        return SplineTransformer(n_knots=trial.suggest_int("ST_n_knots", 2, 8),
                                 degree=trial.suggest_int("ST_degree", 1, 4))
    elif preprocessor == "PolynomialFeatures":
        return PolynomialFeatures(degree=trial.suggest_int("PF_degree", 1, 2), include_bias=False)
    elif preprocessor == "KBinsDiscretizer":
        return KBinsDiscretizer(n_bins=trial.suggest_int("KBins_n_bins", 2, 10), encode="ordinal", strategy="uniform")

    # Feature scaling
    elif preprocessor == "StandardScaler":
        return StandardScaler()
    elif preprocessor == "MinMaxScaler":
        return MinMaxScaler()
    elif preprocessor == "RobustScaler":
        return RobustScaler()
    elif preprocessor == "PowerTransformer":
        return PowerTransformer()
    elif preprocessor == "QuantileTransformer":
        return QuantileTransformer(n_quantiles=trial.suggest_int("QT_n_quantiles", 5, 1000))

    else:
        raise ValueError("Preprocessor not implemented")


class EmptyStep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
