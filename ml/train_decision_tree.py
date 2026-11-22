"""
ml/train_decision_tree.py

Trains a DecisionTreeClassifier on ml/dataset.csv to recommend a planning
type (Compact / Standard / Premium / Luxury) from plot and requirement
features, per the brief's section 5. Saves the trained model + label
encoder with joblib so the FastAPI backend can load it without retraining
on every request.

Run:
    python ml/train_decision_tree.py
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset.csv")
MODEL_PATH = os.path.join(HERE, "decision_tree_model.pkl")

FEATURES = [
    "plot_area", "plot_length", "plot_width",
    "bedrooms", "bathrooms", "floors",
    "budget", "parking_required", "family_size",
]
TARGET = "planning_type"


def train():
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES]
    y_raw = df[TARGET]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Decision tree test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    joblib.dump({"model": clf, "encoder": encoder, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return clf, encoder, acc


if __name__ == "__main__":
    train()
