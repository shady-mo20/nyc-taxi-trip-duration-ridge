"""
Feature engineering for NYC Taxi Trip Duration.

Important:
- We do NOT use dropoff_datetime as a model feature.
- dropoff_datetime leaks the answer because:
      dropoff_datetime - pickup_datetime = trip_duration
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NYC_CENTER_LAT = 40.7580
NYC_CENTER_LON = -73.9855


NUMERIC_FEATURES = [
    "passenger_count",

    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_minute",

    "is_weekend",
    "is_rush_hour",
    "is_night",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",

    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",

    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",

    "haversine_km",
    "manhattan_km",
    "log_haversine_km",
    "sqrt_haversine_km",
    "haversine_km_squared",
    "log_manhattan_km",
    "sqrt_manhattan_km",

    "bearing",
    "bearing_sin",
    "bearing_cos",

    "center_latitude",
    "center_longitude",

    "pickup_distance_to_center",
    "dropoff_distance_to_center",

    "pickup_jfk",
    "dropoff_jfk",
    "pickup_lga",
    "dropoff_lga",
    "pickup_ewr",
    "dropoff_ewr",

    "distance_x_rush",
    "distance_x_weekend",
    "distance_x_night",
    "distance_x_hour",

    # Added by target encoding.
    "pickup_cell_te",
    "dropoff_cell_te",
    "route_cell_te",
    "hour_dow_te",
    "vendor_hour_te",
]

CATEGORICAL_FEATURES = [
    "vendor_id",
    "store_and_fwd_flag",
]

TARGET_ENCODING_COLUMNS = [
    "pickup_cell",
    "dropoff_cell",
    "route_cell",
    "hour_dow",
    "vendor_hour",
]


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def bearing_degrees(lat1, lon1, lat2, lon2):
    """Vectorized bearing/direction from pickup to dropoff in degrees."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def make_cell(lat: pd.Series, lon: pd.Series, precision: int = 100) -> pd.Series:
    """
    Convert coordinates to a coarse grid cell string.

    precision=100 roughly groups coordinates by 0.01 degrees.
    """
    lat_bin = np.floor(lat * precision)
    lon_bin = np.floor(lon * precision)

    lat_bin = pd.Series(lat_bin, index=lat.index).fillna(-999999).astype(int).astype(str)
    lon_bin = pd.Series(lon_bin, index=lon.index).fillna(-999999).astype(int).astype(str)

    return lat_bin + "_" + lon_bin


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all model features from the raw taxi data."""
    df = df.copy()

    required_columns = [
        "pickup_datetime",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "vendor_id",
        "store_and_fwd_flag",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Time features from pickup only.
    pickup_dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["pickup_month"] = pickup_dt.dt.month
    df["pickup_day"] = pickup_dt.dt.day
    df["pickup_hour"] = pickup_dt.dt.hour
    df["pickup_dayofweek"] = pickup_dt.dt.dayofweek
    df["pickup_minute"] = pickup_dt.dt.minute

    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

    morning_rush = df["pickup_hour"].between(7, 10)
    evening_rush = df["pickup_hour"].between(16, 19)
    df["is_rush_hour"] = (morning_rush | evening_rush).astype(int)

    df["is_night"] = ((df["pickup_hour"] >= 22) | (df["pickup_hour"] <= 5)).astype(int)

    # Cyclical time representation.
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)

    # Coordinate difference features.
    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["abs_lat_diff"] = df["lat_diff"].abs()
    df["abs_lon_diff"] = df["lon_diff"].abs()

    # Distance features.
    df["haversine_km"] = haversine_distance_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["manhattan_km"] = (
        haversine_distance_km(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["pickup_latitude"],
            df["dropoff_longitude"],
        )
        + haversine_distance_km(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["pickup_longitude"],
        )
    )

    # Transform distance to help a linear model.
    df["log_haversine_km"] = np.log1p(df["haversine_km"])
    df["sqrt_haversine_km"] = np.sqrt(np.maximum(df["haversine_km"], 0))
    df["haversine_km_squared"] = df["haversine_km"] ** 2

    df["log_manhattan_km"] = np.log1p(df["manhattan_km"])
    df["sqrt_manhattan_km"] = np.sqrt(np.maximum(df["manhattan_km"], 0))

    # Direction.
    df["bearing"] = bearing_degrees(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["bearing_sin"] = np.sin(2 * np.pi * df["bearing"] / 360)
    df["bearing_cos"] = np.cos(2 * np.pi * df["bearing"] / 360)

    # Midpoint of route.
    df["center_latitude"] = (df["pickup_latitude"] + df["dropoff_latitude"]) / 2
    df["center_longitude"] = (df["pickup_longitude"] + df["dropoff_longitude"]) / 2

    # Distance to approximate NYC center.
    df["pickup_distance_to_center"] = haversine_distance_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        NYC_CENTER_LAT,
        NYC_CENTER_LON,
    )
    df["dropoff_distance_to_center"] = haversine_distance_km(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        NYC_CENTER_LAT,
        NYC_CENTER_LON,
    )

    # Airport flags using approximate coordinate boxes.
    df["pickup_jfk"] = (
        df["pickup_latitude"].between(40.62, 40.67)
        & df["pickup_longitude"].between(-73.82, -73.75)
    ).astype(int)
    df["dropoff_jfk"] = (
        df["dropoff_latitude"].between(40.62, 40.67)
        & df["dropoff_longitude"].between(-73.82, -73.75)
    ).astype(int)

    df["pickup_lga"] = (
        df["pickup_latitude"].between(40.75, 40.79)
        & df["pickup_longitude"].between(-73.90, -73.85)
    ).astype(int)
    df["dropoff_lga"] = (
        df["dropoff_latitude"].between(40.75, 40.79)
        & df["dropoff_longitude"].between(-73.90, -73.85)
    ).astype(int)

    df["pickup_ewr"] = (
        df["pickup_latitude"].between(40.66, 40.72)
        & df["pickup_longitude"].between(-74.20, -74.14)
    ).astype(int)
    df["dropoff_ewr"] = (
        df["dropoff_latitude"].between(40.66, 40.72)
        & df["dropoff_longitude"].between(-74.20, -74.14)
    ).astype(int)

    # Grid and interaction categorical features for target encoding.
    df["pickup_cell"] = make_cell(df["pickup_latitude"], df["pickup_longitude"], precision=100)
    df["dropoff_cell"] = make_cell(df["dropoff_latitude"], df["dropoff_longitude"], precision=100)
    df["route_cell"] = df["pickup_cell"] + "__" + df["dropoff_cell"]

    df["hour_dow"] = df["pickup_hour"].astype(str) + "_" + df["pickup_dayofweek"].astype(str)
    df["vendor_hour"] = df["vendor_id"].astype(str) + "_" + df["pickup_hour"].astype(str)

    # Manual interactions useful for Ridge.
    df["distance_x_rush"] = df["haversine_km"] * df["is_rush_hour"]
    df["distance_x_weekend"] = df["haversine_km"] * df["is_weekend"]
    df["distance_x_night"] = df["haversine_km"] * df["is_night"]
    df["distance_x_hour"] = df["haversine_km"] * df["pickup_hour"]

    return df


def get_cleaning_thresholds(train_features: pd.DataFrame) -> dict:
    """Compute thresholds from training data only."""
    return {
        "duration_low": float(train_features["trip_duration"].quantile(0.001)),
        "duration_high": float(train_features["trip_duration"].quantile(0.999)),
        "distance_low": float(train_features["haversine_km"].quantile(0.001)),
        "distance_high": float(train_features["haversine_km"].quantile(0.999)),
    }


def build_clean_mask(df: pd.DataFrame, thresholds: dict, require_target: bool = True) -> pd.Series:
    """Create a conservative cleaning mask for training/validation data."""
    mask = (
        df["haversine_km"].between(thresholds["distance_low"], thresholds["distance_high"])
        & df["pickup_longitude"].between(-75, -72)
        & df["dropoff_longitude"].between(-75, -72)
        & df["pickup_latitude"].between(40, 42)
        & df["dropoff_latitude"].between(40, 42)
        & (df["passenger_count"] > 0)
    )

    if require_target:
        mask = mask & df["trip_duration"].between(
            thresholds["duration_low"],
            thresholds["duration_high"],
        )

    return mask
