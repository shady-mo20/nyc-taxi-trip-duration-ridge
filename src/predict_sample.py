"""
Load a saved model and predict a CSV file.

Run from ~/Desktop/ml:

Predict the small sample test:
    python3 src/predict_sample.py --input split_sample/test.csv --output reports/sample_test_predictions.csv

Evaluate sample validation if it has trip_duration:
    python3 src/predict_sample.py --input split_sample/val.csv --output reports/sample_val_predictions.csv

Predict professor test:
    python3 src/predict_sample.py --input split/test.csv --output reports/test_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

from config import DEFAULT_MODEL_PATH, PROJECT_ROOT
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict taxi trip duration from a CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved model bundle path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "predictions.csv",
        help="Output predictions CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Reading input: {args.input}")
    df = pd.read_csv(args.input)

    print(f"Loading model: {args.model}")
    bundle = joblib.load(args.model)

    model = bundle["model"]
    target_encoder = bundle["target_encoder"]

    print("Creating features...")
    features_df = add_features(df)

    print("Applying target encoder...")
    encoded_df = target_encoder.transform(features_df)

    final_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = encoded_df[final_features]

    print("Predicting...")
    pred_log = model.predict(X)
    pred_seconds = np.maximum(np.expm1(pred_log), 1)

    output = pd.DataFrame()

    if "id" in df.columns:
        output["id"] = df["id"]
    else:
        output["row_id"] = np.arange(len(df))

    output["trip_duration"] = pred_seconds

    if "trip_duration" in df.columns:
        y_true_log = np.log1p(df["trip_duration"])
        r2_log = r2_score(y_true_log, pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_true_log, pred_log))

        y_true_seconds = df["trip_duration"]
        r2_seconds = r2_score(y_true_seconds, pred_seconds)

        output["actual_trip_duration"] = df["trip_duration"]
        output["predicted_log_trip_duration"] = pred_log
        output["actual_log_trip_duration"] = y_true_log

        print("\nEvaluation because input contains trip_duration:")
        print(f"  R2 on log target: {r2_log:.5f}")
        print(f"  RMSE on log target: {rmse_log:.5f}")
        print(f"  R2 on seconds: {r2_seconds:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"\nPredictions saved to: {args.output}")
    print(output.head())


if __name__ == "__main__":
    main()
