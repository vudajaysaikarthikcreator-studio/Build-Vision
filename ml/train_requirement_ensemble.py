"""
ml/train_requirement_ensemble.py

The brief specifies a Decision Tree for requirement analysis
(train_decision_tree.py) — kept as the PRIMARY model because a single
tree is directly inspectable (you can print/plot the exact split rules,
which matters for a viva). This script trains a GradientBoostingClassifier
on the identical dataset/features as a CHALLENGER model: gradient-boosted
trees are consistently among the strongest performers on small/medium
tabular data (the same family XGBoost/LightGBM belong to), so this is the
"best available" upgrade path that doesn't require any package beyond
scikit-learn (already a dependency).

If you later have internet access and want to go one step further,
swapping GradientBoostingClassifier for xgboost.XGBClassifier or
lightgbm.LGBMClassifier is a near drop-in change (same .fit/.predict/
.predict_proba interface) — not done here because this environment has
no network access to install/verify them.

Run:
    python ml/train_requirement_ensemble.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "dataset.csv")
MODEL_PATH = os.path.join(HERE, "requirement_ensemble_model.pkl")

FEATURES = [
    "plot_area", "plot_length", "plot_width",
    "bedrooms", "bathrooms", "floors",
    "budget", "parking_required", "family_size",
]
TARGET = "planning_type"


def train():
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Gradient Boosting (challenger) test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=encoder.classes_))

    joblib.dump({"model": clf, "encoder": encoder, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return clf, encoder, acc


if __name__ == "__main__":
    train()
