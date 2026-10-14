
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "training_snapshots.csv"
OUTPUT_PATH = ROOT / "reports" / "customer_level_evaluation.csv"

FEATURES = [
    "TotalSpend",
    "TotalOrders",
    "TotalItems",
    "UniqueProducts",
    "RecencyDays",
    "CustomerTenureDays",
    "AverageOrderValue",
]


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, scores),
        "Average_Precision": average_precision_score(
            y_test, scores
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["CustomerID", "Churn"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=FEATURES + ["CustomerID", "Churn"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["Churn"] = df["Churn"].astype(int)

    # Split by customer ID, not by individual rows.
    # All snapshots belonging to one customer stay together.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df[FEATURES],
            df["Churn"],
            groups=df["CustomerID"],
        )
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    train_ids = set(train["CustomerID"])
    test_ids = set(test["CustomerID"])

    assert train_ids.isdisjoint(test_ids), (
        "Customer leakage detected: train and test IDs overlap."
    )

    X_train = train[FEATURES]
    y_train = train["Churn"]

    X_test = test[FEATURES]
    y_test = test["Churn"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both classes must exist in train and test data."
        )

    results = []

    models = [
        (
            "Majority-class baseline",
            DummyClassifier(strategy="prior"),
        ),
        (
            "Recency-only Logistic Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000),
            ),
        ),
        (
            "Full-feature Logistic Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000),
            ),
        ),
        (
            "Full-feature Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]

    for name, model in models:
        features = (
            ["RecencyDays"]
            if name == "Recency-only Logistic Regression"
            else FEATURES
        )

        results.append(
            evaluate(
                name,
                model,
                train[features],
                y_train,
                test[features],
                y_test,
            )
        )

    comparison = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print("CUSTOMER-LEVEL EVALUATION")
    print("-------------------------")
    print(f"Training records: {len(train):,}")
    print(f"Test records: {len(test):,}")
    print(f"Training customers: {len(train_ids):,}")
    print(f"Test customers: {len(test_ids):,}")
    print(f"Customer overlap: {len(train_ids & test_ids)}")
    print(f"Test churn rate: {y_test.mean():.3f}")

    print("\nMODEL COMPARISON")
    print(
        comparison.to_string(
            index=False,
            formatters={
                column: "{:.3f}".format
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "ROC_AUC",
                    "Average_Precision",
                ]
            },
        )
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()