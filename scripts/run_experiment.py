from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import confusion_df, evaluate_predictions, save_confusion_plot
from src.features import fit_transform_features
from src.io_utils import ensure_dir, load_csv, save_csv, save_json
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run replication experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = cfg["target_column"]
    use_temporal = bool(cfg.get("use_temporal_split", False))
    date_col = cfg.get("date_column")
    test_size = float(cfg.get("test_size", 0.2))
    seed = int(cfg.get("seed", 42))

    if use_temporal and date_col in df.columns:
        sorted_df = df.sort_values(by=date_col).reset_index(drop=True)
        split_idx = int(len(sorted_df) * (1 - test_size))
        train_df = sorted_df.iloc[:split_idx].copy()
        test_df = sorted_df.iloc[split_idx:].copy()
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[target],
        )
    return train_df, test_df


def run_single_setting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    setting_name: str,
    use_smote: bool,
    output_dir: Path,
) -> None:
    target_col = cfg["target_column"]
    seed = int(cfg.get("seed", 42))
    top_k = int(cfg.get("top_k_features", 20))
    outlier_cap = float(cfg.get("outlier_cap", 0.95))
    run_cv = bool(cfg.get("run_cv", True))
    cv_folds = int(cfg.get("cv_folds", 5))

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].to_numpy()
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].to_numpy()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    labels = label_encoder.classes_.tolist()

    Xt_train, Xt_test, feat_artifacts = fit_transform_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train_encoded,
        seed=seed,
        top_k_features=top_k,
        outlier_cap=outlier_cap,
    )

    if use_smote:
        smote = SMOTE(random_state=seed)
        Xt_train, y_train_encoded = smote.fit_resample(Xt_train, y_train_encoded)

    selected_features_payload = {"selected_features": feat_artifacts.feature_names}
    save_json(output_dir / setting_name / "selected_features.json", selected_features_payload)

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        model_params = model_cfg.get("params", {})
        model = build_model(model_name=model_name, seed=seed, params=model_params)

        model.fit(Xt_train, y_train_encoded)
        y_pred_encoded = model.predict(Xt_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        y_proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xt_test)
            y_proba = np.asarray(proba)

        metrics = evaluate_predictions(y_test, y_pred, y_proba, labels=labels)
        if run_cv:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            cv_scores = cross_val_score(model, Xt_train, y_train_encoded, cv=cv, scoring="accuracy")
            metrics["cv_accuracy_mean"] = float(np.mean(cv_scores))
            metrics["cv_accuracy_std"] = float(np.std(cv_scores))

        model_dir = output_dir / setting_name / model_name
        ensure_dir(model_dir)
        save_json(model_dir / "metrics.json", metrics)

        cm_df = confusion_df(y_test, y_pred, labels=labels)
        save_csv(model_dir / "confusion_matrix.csv", cm_df.reset_index().rename(columns={"index": "actual"}))
        save_confusion_plot(
            cm_df,
            model_dir / "confusion_matrix.png",
            title=f"{model_name} ({setting_name})",
        )


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    dataset_path = Path(cfg["dataset_path"])

    df = load_csv(dataset_path)
    if cfg["target_column"] not in df.columns:
        raise KeyError(f"Target column '{cfg['target_column']}' is missing from dataset.")

    train_df, test_df = split_data(df, cfg)

    output_root = Path(cfg.get("output_root", "results"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment_name", "replication")
    run_dir = output_root / f"{exp_name}_{run_id}"
    ensure_dir(run_dir)
    save_json(run_dir / "config_snapshot.json", cfg)

    if cfg["settings"].get("run_imbalanced", True):
        run_single_setting(
            train_df=train_df,
            test_df=test_df,
            cfg=cfg,
            setting_name="imbalanced",
            use_smote=False,
            output_dir=run_dir,
        )
    if cfg["settings"].get("run_smote", True):
        run_single_setting(
            train_df=train_df,
            test_df=test_df,
            cfg=cfg,
            setting_name="smote",
            use_smote=True,
            output_dir=run_dir,
        )

    print(f"Experiment finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
