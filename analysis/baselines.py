"""Context for the notebook's headline numbers.

About 88% of customers did not subscribe, so a model that always answers "no" already scores
about 88% accuracy. This script measures that baseline, compares minority-class metrics, and
checks how much the models depend on `duration`, which is only known after a call has ended.

Run: python analysis/baselines.py data/bank-full.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


def load(path: str) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, sep=";")
    return data.drop(columns=["y"]), (data["y"] == "yes").astype(int)


def preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )


def score(model, x_test, y_test) -> dict:
    predicted = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "roc_auc": round(roc_auc_score(y_test, probability), 4),
        "pr_auc": round(average_precision_score(y_test, probability), 4),
        "yes_recall": round(recall_score(y_test, predicted, zero_division=0), 4),
        "yes_f1": round(f1_score(y_test, predicted, zero_division=0), 4),
    }


def run(path: str) -> dict:
    features, target = load(path)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED
    )
    results = {
        "rows": len(features),
        "positive_rate": round(float(target.mean()), 4),
        "split": "80/20 stratified, seed 42 (differs from the notebook's 60/20/20 randomSplit)",
    }
    majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    results["always_no"] = score(majority, x_test, y_test)

    configurations = {"with_duration": x_train.columns.tolist(),
                      "without_duration": [c for c in x_train.columns if c != "duration"]}
    for name, columns in configurations.items():
        for label, estimator in (
            ("logistic_regression", LogisticRegression(max_iter=2000)),
            ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1)),
        ):
            model = Pipeline(
                [("prepare", preprocessor(x_train[columns])), ("model", estimator)]
            ).fit(x_train[columns], y_train)
            results[f"{label}_{name}"] = score(model, x_test[columns], y_test)
    return results


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "data/bank-full.csv"
    output = run(source)
    Path(__file__).with_name("baseline_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
