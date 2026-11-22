"""
ml/train_random_forest.py

Trains a RandomForestRegressor on ml/cost_dataset.csv to approximate
construction cost from plot area, built-up area, floors, room counts and
construction category, per the brief's section 10. Saves the trained
model with joblib.

Run:
    python ml/train_random_forest.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "cost_dataset.csv")
MODEL_PATH = os.path.join(HERE, "cost_model.pkl")

NUMERIC_FEATURES = ["plot_area", "builtup_area", "floors", "num_rooms", "num_bathrooms"]
CATEGORICAL_FEATURES = ["category"]
TARGET = "cost"


def train():
    df = pd.read_csv(DATASET_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Random forest cost model — MAE: Rs.{mae:,.0f}   R2: {r2:.3f}")

    joblib.dump({
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return pipeline, r2


if __name__ == "__main__":
    train()
