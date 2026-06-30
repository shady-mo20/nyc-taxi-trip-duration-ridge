# NYC Taxi Trip Duration Prediction

## Overview

This project predicts New York City taxi trip duration using a complete machine learning regression pipeline.

The model is fixed to Ridge Regression with alpha=1, and the target variable is transformed using:

np.log1p(trip_duration)

This transformation is used because trip duration is highly right-skewed and contains extreme outliers.

## Problem Statement

Given taxi trip information such as pickup time, pickup/dropoff coordinates, passenger count, vendor ID, and store-and-forward flag, the goal is to predict the total trip duration in seconds.

This is a supervised regression problem.

## Project Structure

src/
- config.py
- features.py
- target_encoder.py
- train_model.py
- predict_sample.py

models/
- stores generated trained models locally

reports/
- stores generated metrics and prediction files locally

## Ignored Files

The following files are not uploaded to GitHub:

- .venv/
- split/
- split_sample/
- generated CSV prediction files
- reports/*.csv
- *.zip

These files are local datasets, generated prediction reports, zip files, or environment files. The trained Ridge model is included because its size is small and useful for reproducibility.

## Installation

Create and activate virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

## How to Run

Train on sample data:

python3 src/train_model.py --data-dir split_sample

Train on full data:

python3 src/train_model.py --data-dir split

Train final model on train + validation:

python3 src/train_model.py --data-dir split --save-final-trained-on-train-val

Predict on test data:

python3 src/predict_sample.py --input split/test.csv --output reports/test_predictions.csv

## Feature Engineering

The model uses several feature groups:

Time features:
- pickup_month
- pickup_day
- pickup_hour
- pickup_dayofweek
- pickup_minute
- is_weekend
- is_rush_hour
- is_night

Distance and location features:
- haversine_km
- manhattan_km
- log_haversine_km
- log_manhattan_km
- latitude and longitude differences
- bearing
- trip center coordinates

Interaction features:
- distance_x_rush
- distance_x_weekend
- distance_x_night

Categorical features:
- vendor_id
- store_and_fwd_flag

Target encoding features:
- pickup_cell
- dropoff_cell
- route_cell
- hour_dow
- vendor_hour

Target encoding is fitted on training data only to avoid data leakage.

## Model

The final model is:

Ridge(alpha=1)

The pipeline includes:

- SimpleImputer
- StandardScaler
- OneHotEncoder
- Ridge Regression

The model is trained on:

np.log1p(trip_duration)

Predictions are converted back to seconds using:

np.expm1(prediction)

## Results

Evaluation metric:

R2 score on np.log1p(trip_duration)

Training R2 on log target: 0.71609
Validation R2 on log target: 0.71560
Clean Test R2 on log target: 0.71703

The training, validation, and clean test scores are close, which indicates that the model is stable and not overfitting.

## Important Notes

dropoff_datetime was not used as a model feature because it leaks the target duration.

The main improvement came from feature engineering, not from changing the model.
