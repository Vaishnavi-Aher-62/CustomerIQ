
from pathlib import Path

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "processed" / "online_retail_clean.csv"
OUTPUT_PATH = ROOT / "reports" / "churn_sensitivity.csv"

HORIZONS = [30, 60, 90, 120]

TRAIN_CUTOFFS = [
    "2011-03-01",
    "2011-04-01",
    "2011-05-01",
    "2011-06-01",
]

# Use a common evaluation date for all four horizons.
TEST_CUTOFF = "2011-08-01"

FEATURES = [
    "TotalSpend",
    "TotalOrders",
    "TotalItems",
    "UniqueProducts",
    "RecencyDays",
    "CustomerTenureDays",
    "AverageOrderValue",
]


def build_snapshot(data, cutoff_string, horizon_days):
    cutoff = pd.Timestamp(cutoff_string)
    future_end = cutoff + pd.Timedelta(days=horizon_days)

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

    future_customers = set(future["CustomerID"].unique())

    features["Churn"] = (
        ~features.index.isin(future_customers)
    ).astype(int)

    features = features.drop(
        columns=["FirstPurchase", "LastPurchase"]
    )

    features = features.reset_index()
    features["SnapshotDate"] = cutoff

    return features


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["InvoiceDate"],
    )

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

    # Match the original pipeline's valid-purchase rules.
    df = df[
        ~df["InvoiceNo"].str.upper().str.startswith("C")
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & df["CustomerID"].notna()
    ].copy()

    df["CustomerID"] = (
        df["CustomerID"]
        .astype(float)
        .astype("int64")
        .astype(str)
    )

    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    data_end = df["InvoiceDate"].max()
    test_date = pd.Timestamp(TEST_CUTOFF)

    results = []

    for horizon in HORIZONS:
        future_end = test_date + pd.Timedelta(days=horizon)

        if future_end > data_end:
            print(
                f"Skipping {horizon} days: insufficient "
                f"future data after {TEST_CUTOFF}."
            )
            continue

        test = build_snapshot(
            df, TEST_CUTOFF, horizon
        )

        # Only use training snapshots whose entire label
        # window ends on or before the test snapshot date.
        safe_cutoffs = [
            cutoff
            for cutoff in TRAIN_CUTOFFS
            if (
                pd.Timestamp(cutoff)
                + pd.Timedelta(days=horizon)
            ) <= test_date
        ]

        train_frames = [
            build_snapshot(df, cutoff, horizon)
            for cutoff in safe_cutoffs
        ]

        if not train_frames:
            print(
                f"Skipping {horizon} days: no safe "
                "training snapshots."
            )
            continue

        train = pd.concat(
            train_frames,
            ignore_index=True,
        )

        X_train = train[FEATURES]
        y_train = train["Churn"].astype(int)

        X_test = test[FEATURES]
        y_test = test["Churn"].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(
                f"Skipping {horizon} days: "
                "only one class in train or test."
            )
            continue

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )

        model.fit(X_train, y_train)

        scores = model.predict_proba(X_test)[:, 1]

        results.append({
            "HorizonDays": horizon,
            "TestSnapshot": TEST_CUTOFF,
            "TrainingSnapshots": len(safe_cutoffs),
            "TrainingRows": len(train),
            "TestCustomers": test["CustomerID"].nunique(),
            "ChurnRate": y_test.mean(),
            "ROC_AUC": roc_auc_score(y_test, scores),
            "Average_Precision": average_precision_score(
                y_test, scores
            ),
        })

    if not results:
        raise RuntimeError(
            "No valid sensitivity results were produced."
        )

    report = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(OUTPUT_PATH, index=False)

    print("\nCHURN-DEFINITION SENSITIVITY")
    print("----------------------------")

    print(
        report.to_string(
            index=False,
            formatters={
                "ChurnRate": "{:.3f}".format,
                "ROC_AUC": "{:.3f}".format,
                "Average_Precision": "{:.3f}".format,
            },
        )
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
