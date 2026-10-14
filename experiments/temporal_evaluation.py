
from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "online_retail_clean.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_DAYS = 60
TRAIN_CUTOFF = pd.Timestamp("2011-06-01")
TEST_CUTOFF = pd.Timestamp("2011-09-01")

df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

# Exclude cancellations and invalid purchase rows
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

def make_snapshot(data, cutoff):
    """Build historical features and labels for one cutoff date."""
    history = data[data["InvoiceDate"] < cutoff].copy()

    future_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
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
        LastPurchase=("InvoiceDate", "max")
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

    return features.drop(
        columns=["FirstPurchase", "LastPurchase"]
    )

# Train labels: June 1 through July 31.
# Test labels: September 1 through October 31.
train = make_snapshot(df, TRAIN_CUTOFF)
test = make_snapshot(df, TEST_CUTOFF)

feature_cols = [
    "TotalSpend",
    "TotalOrders",
    "TotalItems",
    "UniqueProducts",
    "RecencyDays",
    "CustomerTenureDays",
    "AverageOrderValue"
]

# The test cohort must contain enough examples of both classes.
print("Training cutoff:", TRAIN_CUTOFF.date())
print("Testing cutoff:", TEST_CUTOFF.date())
print("Training customers:", len(train))
print("Testing customers:", len(test))
print("\nTest churn distribution:")
print(test["Churn"].value_counts())

X_train = train[feature_cols]
y_train = train["Churn"]
X_test = test[feature_cols]
y_test = test["Churn"]

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    auc = roc_auc_score(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Average precision: {ap:.4f}")

    results.append({
        "Model": name,
        "ROC_AUC": auc,
        "AveragePrecision": ap
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / "temporal_model_comparison.csv",
    index=False
)

print("\n--- TEMPORAL MODEL COMPARISON ---")
print(results_df.to_string(index=False))
print("\nSaved temporal evaluation report.")
