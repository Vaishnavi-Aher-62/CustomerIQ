
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "online_retail_clean.csv"
OUTPUT_PATH = ROOT / "reports" / "gap_feature_comparison.csv"

TRAIN_CUTOFFS = [
    "2011-03-01",
    "2011-04-01",
    "2011-05-01",
    "2011-06-01",
]
VALIDATION_CUTOFF = "2011-08-01"
TEST_CUTOFF = "2011-10-01"
HORIZON_DAYS = 60

BASE_FEATURES = [
    "TotalSpend",
    "TotalOrders",
    "TotalItems",
    "UniqueProducts",
    "RecencyDays",
    "CustomerTenureDays",
    "AverageOrderValue",
]

GAP_FEATURES = [
    "AverageDaysBetweenOrders",
    "OrderGapStdDays",
    "RecencyToAvgGapRatio",
]


def build_snapshot(data, cutoff_string):
    cutoff = pd.Timestamp(cutoff_string)
    future_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

    history = data[data["InvoiceDate"] < cutoff]
    future = data[
        (data["InvoiceDate"] >= cutoff)
        & (data["InvoiceDate"] < future_end)
    ]

    features = history.groupby("CustomerID").agg(
        TotalSpend=("TotalAmount", "sum"),
        TotalOrders=("InvoiceNo", "nunique"),
        TotalItems=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
    )

    features["RecencyDays"] = (
        cutoff - features["LastPurchase"]
    ).dt.days

    features["CustomerTenureDays"] = (
        cutoff - features["FirstPurchase"]
    ).dt.days

    features["AverageOrderValue"] = (
        features["TotalSpend"] / features["TotalOrders"]
    )

    # Calculate gaps between distinct purchase dates.
    order_dates = (
        history[["CustomerID", "InvoiceDate"]]
        .drop_duplicates()
        .sort_values(["CustomerID", "InvoiceDate"])
    )

    order_dates["GapDays"] = (
        order_dates.groupby("CustomerID")["InvoiceDate"]
        .diff()
        .dt.total_seconds() / 86400
    )

    gap_stats = order_dates.groupby("CustomerID")["GapDays"].agg(
        AverageDaysBetweenOrders="mean",
        OrderGapStdDays="std",
    )

    features = features.join(gap_stats)

    # Compare current recency with the customer's normal gap.
    features["RecencyToAvgGapRatio"] = (
        features["RecencyDays"]
        / features["AverageDaysBetweenOrders"].replace(0, np.nan)
    )

    future_customers = set(future["CustomerID"].unique())

    features["Churn"] = (
        ~features.index.isin(future_customers)
    ).astype(int)

    features = features.drop(
        columns=["FirstPurchase", "LastPurchase"]
    ).reset_index()

    features["SnapshotDate"] = cutoff
    return features


def main():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["InvoiceDate"],
    )

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & df["CustomerID"].notna()
    ].copy()

    df["CustomerID"] = (
        df["CustomerID"].astype(float).astype("int64").astype(str)
    )
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    data_end = df["InvoiceDate"].max()
    all_cutoffs = TRAIN_CUTOFFS + [VALIDATION_CUTOFF, TEST_CUTOFF]

    for cutoff in all_cutoffs:
        if pd.Timestamp(cutoff) + pd.Timedelta(
            days=HORIZON_DAYS
        ) > data_end:
            raise ValueError(
                f"Insufficient future data for {cutoff}"
            )

    train = pd.concat(
        [build_snapshot(df, cutoff) for cutoff in TRAIN_CUTOFFS],
        ignore_index=True,
    )
    validation = build_snapshot(df, VALIDATION_CUTOFF)
    test = build_snapshot(df, TEST_CUTOFF)

    feature_sets = {
        "Original seven features": BASE_FEATURES,
        "Seven features plus purchase gaps": (
            BASE_FEATURES + GAP_FEATURES
        ),
    }

    results = []

    for name, columns in feature_sets.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )

        model.fit(train[columns], train["Churn"].astype(int))

        # Validation metrics are included for context.
        val_scores = model.predict_proba(
            validation[columns]
        )[:, 1]

        test_scores = model.predict_proba(test[columns])[:, 1]

        results.append({
            "FeatureSet": name,
            "Validation_ROC_AUC": roc_auc_score(
                validation["Churn"], val_scores
            ),
            "Test_ROC_AUC": roc_auc_score(
                test["Churn"], test_scores
            ),
            "Test_Average_Precision": average_precision_score(
                test["Churn"], test_scores
            ),
            "Test_Churn_Rate": test["Churn"].mean(),
            "Test_Customers": test["CustomerID"].nunique(),
        })

    report = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nPURCHASE-GAP FEATURE EXPERIMENT")
    print("--------------------------------")
    print(report.to_string(
        index=False,
        formatters={
            "Validation_ROC_AUC": "{:.3f}".format,
            "Test_ROC_AUC": "{:.3f}".format,
            "Test_Average_Precision": "{:.3f}".format,
            "Test_Churn_Rate": "{:.3f}".format,
        },
    ))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
