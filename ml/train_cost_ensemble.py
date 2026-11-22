"""
ml/train_cost_ensemble.py

The brief specifies Random Forest Regression for cost estimation
(train_random_forest.py) — kept as the PRIMARY model. This script trains
a GradientBoostingRegressor on the same cost_dataset.csv as a CHALLENGER:
boosted trees usually edge out a plain random forest on structured
regression tasks like this one. Same note as train_requirement_ensemble.py
applies re: XGBoost/LightGBM as a further upgrade if you have network
access to install them later.

Run:
    python ml/train_cost_ensemble.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "cost_dataset.csv")
MODEL_PATH = os.path.join(HERE, "cost_ensemble_model.pkl")

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
        ("model", GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.06, random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Gradient Boosting cost model (challenger) — MAE: Rs.{mae:,.0f}   R2: {r2:.3f}")

    joblib.dump({
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return pipeline, r2


if __name__ == "__main__":
    train()
