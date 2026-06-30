"""
Simple smoothed target encoder.

It is fitted only on the training split, then applied to validation/test.
This avoids validation leakage during performance evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SmoothTargetEncoder:
    def __init__(self, columns: list[str], smoothing: float = 50.0):
        self.columns = columns
        self.smoothing = smoothing
        self.global_mean_ = None
        self.maps_ = {}

    def fit(self, X: pd.DataFrame, y) -> "SmoothTargetEncoder":
        self.global_mean_ = float(np.mean(y))

        temp = X[self.columns].copy()
        temp["_target"] = pd.Series(y, index=X.index).astype(float)

        for col in self.columns:
            stats = temp.groupby(col)["_target"].agg(["count", "mean"])
            smooth_values = (
                (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                / (stats["count"] + self.smoothing)
            )
            self.maps_[col] = smooth_values.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("SmoothTargetEncoder must be fitted before transform().")

        X = X.copy()

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found during target encoding.")

            X[col + "_te"] = X[col].map(self.maps_[col]).fillna(self.global_mean_)

        return X

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
