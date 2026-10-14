
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "online_retail_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "customer_churn_features.csv"
)

# Load transactions
df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

# Explicitly exclude cancellation invoices
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]

# Keep valid purchase transactions
df = df[
    (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
    & df["CustomerID"].notna()
].copy()

# Find the end of the available data
data_end = df["InvoiceDate"].max()

# Reserve the last 60 days for churn labelling
snapshot_date = data_end - pd.Timedelta(days=60)

print("Data ends:", data_end)
print("Observation cutoff:", snapshot_date)

# Historical transactions: used to build model features
history = df[df["InvoiceDate"] < snapshot_date].copy()

# Future transactions: used only to create churn labels
future = df[
    (df["InvoiceDate"] >= snapshot_date)
    & (df["InvoiceDate"] <= data_end)
].copy()

# Create customer-level historical features
features = history.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalOrders=("InvoiceNo", "nunique"),
    TotalItems=("Quantity", "sum"),
    UniqueProducts=("StockCode", "nunique"),
    FirstPurchase=("InvoiceDate", "min"),
    LastPurchase=("InvoiceDate", "max")
)

features["RecencyDays"] = (
    snapshot_date - features["LastPurchase"]
).dt.days

features["CustomerTenureDays"] = (
    snapshot_date - features["FirstPurchase"]
).dt.days

features["AverageOrderValue"] = (
    features["TotalSpend"] / features["TotalOrders"]
)

# Identify customers who purchased during the future window
future_customers = set(future["CustomerID"].unique())

# 1 = no purchase in the following 60 days
features["Churn"] = (
    ~features.index.isin(future_customers)
).astype(int)

# Remove raw dates from model features
features = features.drop(
    columns=["FirstPurchase", "LastPurchase"]
)

# Save modelling dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT_PATH, index=True, index_label="CustomerID")

print("\n--- CHURN DATASET SUMMARY ---")
print("Customers:", len(features))
print("\nChurn label counts:")
print(features["Churn"].value_counts())
print("\nChurn label percentages:")
print(
    (features["Churn"].value_counts(normalize=True) * 100)
    .round(2)
)
print("\nFeature columns:")
print(features.columns.tolist())
print("\nSaved to:", OUTPUT_PATH)
