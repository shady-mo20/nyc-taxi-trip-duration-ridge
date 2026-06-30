"""
Configuration for the NYC Taxi Trip Duration project.

Run scripts from the project root:
    ~/Desktop/ml
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = PROJECT_ROOT / "split"
DEFAULT_SAMPLE_DATA_DIR = PROJECT_ROOT / "split_sample"

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ridge_taxi_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "training_metrics.json"
DEFAULT_VAL_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "val_predictions.csv"

RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
