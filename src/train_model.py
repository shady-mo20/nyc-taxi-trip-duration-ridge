"""
Train and save the required Ridge(alpha=1) model.

Expected project structure:
    ~/Desktop/ml/
    ├── split/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── split_sample/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── src/
    ├── models/
    └── reports/

Run from ~/Desktop/ml:
    python3 src/train_model.py --data-dir split

For quick checks:
    python3 src/train_model.py --data-dir split_sample
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DEFAULT_DATA_DIR,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_VAL_PREDICTIONS_PATH,
    RANDOM_STATE,
    RIDGE_ALPHA,
)
from features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_ENCODING_COLUMNS,
    add_features,
    build_clean_mask,
    get_cleaning_thresholds,
)
from target_encoder import SmoothTargetEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ridge model for NYC taxi duration.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.csv and val.csv.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to save metrics JSON.",
    )
    parser.add_argument(
        "--val-predictions-path",
        type=Path,
        default=DEFAULT_VAL_PREDICTIONS_PATH,
        help="Where to save validation predictions CSV.",
    )
    parser.add_argument(
        "--save-final-trained-on-train-val",
        action="store_true",
        help=(
            "After validation evaluation, refit on train+val and save that final model. "
            "Use this when you are ready to predict the test set."
        ),
    )
    return parser.parse_args()


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def evaluate_predictions(y_true_log, y_pred_log) -> dict:
    y_true_seconds = np.expm1(y_true_log)
    y_pred_seconds = np.maximum(np.expm1(y_pred_log), 1)

    return {
        "r2_log_target": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log_target": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_seconds": float(r2_score(y_true_seconds, y_pred_seconds)),
    }


def prepare_xy(df: pd.DataFrame):
    y = np.log1p(df["trip_duration"])
    X = df.drop(columns=["trip_duration", "log_trip_duration"], errors="ignore")
    return X, y


def train_once(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("Creating features...")
    train_features = add_features(train_df)
    val_features = add_features(val_df)

    thresholds = get_cleaning_thresholds(train_features)

    train_mask = build_clean_mask(train_features, thresholds, require_target=True)
    val_mask = build_clean_mask(val_features, thresholds, require_target=True)

    train_clean = train_features.loc[train_mask].copy()
    val_clean = val_features.loc[val_mask].copy()

    print(f"Raw train rows: {len(train_df):,}")
    print(f"Clean train rows: {len(train_clean):,}")
    print(f"Raw validation rows: {len(val_df):,}")
    print(f"Clean validation rows: {len(val_clean):,}")

    X_train, y_train = prepare_xy(train_clean)
    X_val, y_val = prepare_xy(val_clean)

    target_encoder = SmoothTargetEncoder(
        columns=TARGET_ENCODING_COLUMNS,
        smoothing=50.0,
    )

    print("Fitting target encoder on train only...")
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_val_encoded = target_encoder.transform(X_val)

    final_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train_final = X_train_encoded[final_features]
    X_val_final = X_val_encoded[final_features]

    print("Training Ridge(alpha=1)...")
    model = make_model()
    model.fit(X_train_final, y_train)

    train_pred_log = model.predict(X_train_final)
    val_pred_log = model.predict(X_val_final)

    train_metrics = evaluate_predictions(y_train, train_pred_log)
    val_metrics = evaluate_predictions(y_val, val_pred_log)

    val_predictions = pd.DataFrame(
        {
            "id": val_clean["id"].values if "id" in val_clean.columns else np.arange(len(val_clean)),
            "actual_trip_duration": np.expm1(y_val).values,
            "predicted_trip_duration": np.maximum(np.expm1(val_pred_log), 1),
            "actual_log_trip_duration": y_val.values,
            "predicted_log_trip_duration": val_pred_log,
        }
    )

    metadata = {
        "ridge_alpha": RIDGE_ALPHA,
        "target": "np.log1p(trip_duration)",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_encoding_columns": TARGET_ENCODING_COLUMNS,
        "cleaning_thresholds": thresholds,
        "train_rows_raw": int(len(train_df)),
        "train_rows_clean": int(len(train_clean)),
        "validation_rows_raw": int(len(val_df)),
        "validation_rows_clean": int(len(val_clean)),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }

    bundle = {
        "model": model,
        "target_encoder": target_encoder,
        "metadata": metadata,
    }

    return bundle, val_predictions


def refit_on_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame, old_metadata: dict):
    print("Refitting final model on train + validation...")

    full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    full_features = add_features(full_df)

    thresholds = old_metadata["cleaning_thresholds"]
    full_mask = build_clean_mask(full_features, thresholds, require_target=True)
    full_clean = full_features.loc[full_mask].copy()

    X_full, y_full = prepare_xy(full_clean)

    target_encoder = SmoothTargetEncoder(
        columns=TARGET_ENCODING_COLUMNS,
        smoothing=50.0,
    )

    X_full_encoded = target_encoder.fit_transform(X_full, y_full)
    final_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_full_final = X_full_encoded[final_features]

    model = make_model()
    model.fit(X_full_final, y_full)

    metadata = dict(old_metadata)
    metadata["final_model_trained_on"] = "train + validation"
    metadata["final_training_rows_clean"] = int(len(full_clean))

    return {
        "model": model,
        "target_encoder": target_encoder,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / "train.csv"
    val_path = args.data_dir / "val.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train file: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Could not find validation file: {val_path}")

    print(f"Reading train: {train_path}")
    print(f"Reading validation: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    bundle, val_predictions = train_once(train_df, val_df)

    if args.save_final_trained_on_train_val:
        bundle = refit_on_train_val(train_df, val_df, bundle["metadata"])

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.val_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, args.model_path)

    with open(args.metrics_path, "w", encoding="utf-8") as file:
        json.dump(bundle["metadata"], file, indent=2)

    val_predictions.to_csv(args.val_predictions_path, index=False)

    print("\nDone.")
    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.metrics_path}")
    print(f"Validation predictions saved to: {args.val_predictions_path}")

    print("\nTrain metrics:")
    for key, value in bundle["metadata"]["train_metrics"].items():
        print(f"  {key}: {value:.5f}")

    print("\nValidation metrics:")
    for key, value in bundle["metadata"]["validation_metrics"].items():
        print(f"  {key}: {value:.5f}")

    if args.save_final_trained_on_train_val:
        print("\nSaved model was refitted on train + validation for final test prediction.")


if __name__ == "__main__":
    main()
