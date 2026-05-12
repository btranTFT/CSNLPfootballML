from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap numerical values at a configurable quantile."""

    def __init__(self, upper_quantile: float = 0.95):
        self.upper_quantile = upper_quantile
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Winsorizer":
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.upper_bounds_ is None:
            raise RuntimeError("Winsorizer must be fitted before transform().")
        return np.minimum(X, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.asarray(input_features)


@dataclass
class FeatureArtifacts:
    preprocessor: ColumnTransformer
    selector: SelectFromModel
    feature_names: list[str]


def make_preprocessor(
    X: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    outlier_cap: float = 0.95,
) -> ColumnTransformer:
    if numeric_columns is None:
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_columns is None:
        categorical_columns = [c for c in X.columns if c not in numeric_columns]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(upper_quantile=outlier_cap)),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ]
    )


def fit_transform_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    top_k_features: int,
    outlier_cap: float,
) -> tuple[np.ndarray, np.ndarray, FeatureArtifacts]:
    preprocessor = make_preprocessor(X_train, outlier_cap=outlier_cap)
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()
    selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
        threshold=-np.inf,
        max_features=top_k_features,
    )
    Xt_train_sel = selector.fit_transform(Xt_train, y_train)
    Xt_test_sel = selector.transform(Xt_test)

    selected_idx = selector.get_support(indices=True).tolist()
    selected_names = [feature_names[i] for i in selected_idx]
    artifacts = FeatureArtifacts(
        preprocessor=preprocessor,
        selector=selector,
        feature_names=selected_names,
    )
    return Xt_train_sel, Xt_test_sel, artifacts
