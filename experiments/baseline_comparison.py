
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# PATHS AND FEATURES
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT / "data" / "processed" / "training_snapshots.csv"
)

OUTPUT_PATH = ROOT / "reports" / "baseline_comparison.csv"

FEATURES = [
    "TotalSpend",
    "TotalOrders",
    "TotalItems",
    "UniqueProducts",
    "RecencyDays",
    "CustomerTenureDays",
    "AverageOrderValue",
]


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(name, y_true, predictions, scores):
    """Calculate classification and ranking metrics."""

    scores = np.asarray(scores, dtype=float)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": np.nan,
        "Average_Precision": average_precision_score(
            y_true, scores
        ),
    }

    # ROC-AUC is undefined when every observation gets
    # exactly the same score.
    if len(np.unique(scores)) > 1:
        result["ROC_AUC"] = roc_auc_score(
            y_true, scores
        )

    return result


# --------------------------------------------------
# MAIN WORKFLOW
# --------------------------------------------------

def main():

    # 1. Load dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Check that training_snapshots.csv exists."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [
        "CustomerID",
        "SnapshotDate",
        "Split",
        "Churn",
    ]

    missing = sorted(
        set(required_columns) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # 2. Use existing temporal splits
    train = df[df["Split"] == "train"].copy()
    validation = df[
        df["Split"] == "validation"
    ].copy()
    test = df[df["Split"] == "test"].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Train, validation, and test splits "
            "must all contain records."
        )

    # 3. Check missing values and labels
    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if frame[FEATURES + ["Churn"]].isna().any().any():
            raise ValueError(
                f"Missing feature or label values in {name}."
            )

        labels = set(frame["Churn"].astype(int).unique())

        if not labels.issubset({0, 1}):
            raise ValueError(
                f"Churn labels in {name} must be 0 or 1."
            )

        if labels != {0, 1}:
            raise ValueError(
                f"The {name} split must contain both classes."
            )

    X_train = train[FEATURES].astype(float)
    y_train = train["Churn"].astype(int)

    X_test = test[FEATURES].astype(float)
    y_test = test["Churn"].astype(int)

    results = []

    # --------------------------------------------------
    # MODEL 1: MAJORITY-CLASS BASELINE
    # --------------------------------------------------

    majority = DummyClassifier(
        strategy="most_frequent"
    )

    majority.fit(X_train, y_train)

    majority_predictions = majority.predict(X_test)

    # Constant scores provide no ranking information.
    # Their Average Precision equals the test prevalence.
    majority_scores = np.full(
        len(test), float(y_train.mean())
    )

    results.append(
        evaluate_model(
            "Majority-class baseline",
            y_test,
            majority_predictions,
            majority_scores,
        )
    )

    # --------------------------------------------------
    # MODEL 2: RECENCY-ONLY LOGISTIC REGRESSION
    # --------------------------------------------------

    recency_columns = ["RecencyDays"]

    recency_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    recency_model.fit(
        train[recency_columns],
        y_train,
    )

    recency_scores = recency_model.predict_proba(
        test[recency_columns]
    )[:, 1]

    recency_predictions = (
        recency_scores >= 0.5
    ).astype(int)

    results.append(
        evaluate_model(
            "Recency-only Logistic Regression",
            y_test,
            recency_predictions,
            recency_scores,
        )
    )

    # --------------------------------------------------
    # MODEL 3: ONE-RULE RECENCY BASELINE
    # --------------------------------------------------

    # Choose the threshold using validation data only.
    best_threshold = None
    best_validation_f1 = -1.0

    validation_recency = (
        validation["RecencyDays"].astype(float)
    )

    validation_labels = validation["Churn"].astype(int)

    for threshold in sorted(
        validation_recency.unique()
    ):

        validation_predictions = (
            validation_recency > threshold
        ).astype(int)

        current_f1 = f1_score(
            validation_labels,
            validation_predictions,
            zero_division=0,
        )

        if current_f1 > best_validation_f1:
            best_validation_f1 = current_f1
            best_threshold = float(threshold)

    rule_scores = (
        test["RecencyDays"].astype(float)
        > best_threshold
    ).astype(float)

    rule_predictions = rule_scores.astype(int)

    results.append(
        evaluate_model(
            f"Recency rule (> {best_threshold:g} days)",
            y_test,
            rule_predictions,
            rule_scores,
        )
    )

    # --------------------------------------------------
    # MODEL 4: FULL-FEATURE LOGISTIC REGRESSION
    # --------------------------------------------------

    full_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    full_lr.fit(X_train, y_train)

    lr_scores = full_lr.predict_proba(X_test)[:, 1]

    lr_predictions = (
        lr_scores >= 0.5
    ).astype(int)

    results.append(
        evaluate_model(
            "Full-feature Logistic Regression",
            y_test,
            lr_predictions,
            lr_scores,
        )
    )

    # --------------------------------------------------
    # MODEL 5: FULL-FEATURE RANDOM FOREST
    # --------------------------------------------------

    forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    forest.fit(X_train, y_train)

    forest_scores = forest.predict_proba(X_test)[:, 1]

    forest_predictions = (
        forest_scores >= 0.5
    ).astype(int)

    results.append(
        evaluate_model(
            "Full-feature Random Forest",
            y_test,
            forest_predictions,
            forest_scores,
        )
    )

    # --------------------------------------------------
    # SAVE AND DISPLAY RESULTS
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n========== TEST SET SUMMARY ==========")

    print(f"Test records: {len(test):,}")

    print(
        "Unique test customers: "
        f"{test['CustomerID'].nunique():,}"
    )

    print(
        f"Observed test churn rate: {y_test.mean():.3f}"
    )

    print(
        "\nRecency threshold selected on validation data: "
        f"{best_threshold:g} days"
    )

    print(
        "Validation F1 for the recency rule: "
        f"{best_validation_f1:.3f}"
    )

    print("\n========== MODEL COMPARISON ==========")

    print(
        comparison.to_string(
            index=False,
            formatters={
                column: (
                    lambda value:
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.3f}"
                )
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC_AUC",
                    "Average_Precision",
                ]
            },
        )
    )

    print(
        f"\nResults saved to:\n{OUTPUT_PATH}"
    )

    print(
        "\nNote: these are fresh comparison models. "
        "Your saved production model and dashboard "
        "have not been modified."
    )


if __name__ == "__main__":
    main()