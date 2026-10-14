
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])

print("\n--- TRANSACTION OVERVIEW ---")
print("Rows:", len(df))
print("Unique customers:", df["CustomerID"].nunique())
print("Unique invoices:", df["InvoiceNo"].nunique())

print("\n--- CUSTOMER SPENDING ---")
customer_summary = df.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalOrders=("InvoiceNo", "nunique"),
    TotalItems=("Quantity", "sum"),
    FirstPurchase=("InvoiceDate", "min"),
    LastPurchase=("InvoiceDate", "max")
)

customer_summary["DaysBetweenPurchases"] = (
    customer_summary["LastPurchase"]
    - customer_summary["FirstPurchase"]
).dt.days

print(customer_summary.describe().round(2))

print("\n--- TOP 10 CUSTOMERS BY SPENDING ---")
print(
    customer_summary.sort_values("TotalSpend", ascending=False)
    .head(10)
    .round(2)
)

print("\n--- TOP 10 PRODUCTS ---")
print(
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
