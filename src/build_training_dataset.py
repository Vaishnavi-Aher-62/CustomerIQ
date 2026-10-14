
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "data" / "processed" / "online_retail_clean.csv"
)
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "training_snapshots.csv"
)

HORIZON_DAYS = 60

# Earlier snapshots for training, a later validation snapshot,
# and a final untouched test snapshot.
TRAIN_CUTOFFS = [
    "2011-03-01",
    "2011-04-01",
    "2011-05-01",
    "2011-06-01",
]

VALIDATION_CUTOFF = "2011-08-01"
TEST_CUTOFF = "2011-10-01"

df = pd.read_csv(INPUT_PATH, parse_dates=["InvoiceDate"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()

# Keep valid purchases; exclude cancellation invoices.
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
print("Last available transaction:", data_end)

def build_snapshot(data, cutoff_string):
    cutoff = pd.Timestamp(cutoff_string)
    future_end = cutoff + pd.Timedelta(days=HORIZON_DAYS)

    if future_end > data_end:
        raise ValueError(
            f"Insufficient future data for cutoff {cutoff.date()}"
        )

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

# Build the training snapshots
train_frames = [
    build_snapshot(df, cutoff)
    for cutoff in TRAIN_CUTOFFS
]

train = pd.concat(train_frames, ignore_index=True)
train["Split"] = "train"

# Separate validation and final test periods
validation = build_snapshot(df, VALIDATION_CUTOFF)
validation["Split"] = "validation"

test = build_snapshot(df, TEST_CUTOFF)
test["Split"] = "test"

combined = pd.concat(
    [train, validation, test],
    ignore_index=True
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)

print("\n--- SNAPSHOT DATASET SUMMARY ---")
print(combined.groupby("Split").agg(
    Rows=("CustomerID", "size"),
    UniqueCustomers=("CustomerID", "nunique"),
    ChurnRate=("Churn", "mean")
).round(3))

print("\nSnapshot dates:")
print(
    combined.groupby("Split")["SnapshotDate"]
    .agg(["min", "max"])
)

print("\nSaved to:", OUTPUT_PATH)
print("Total snapshot rows:", len(combined))
