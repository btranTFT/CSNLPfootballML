from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier


def build_model(model_name: str, seed: int, params: dict[str, Any] | None = None):
    params = params or {}
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=seed,
            class_weight=params.get("class_weight", None),
            n_jobs=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 150),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "adaboost":
        return AdaBoostClassifier(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.8),
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")
