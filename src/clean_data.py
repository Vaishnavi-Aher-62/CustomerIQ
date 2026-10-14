
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.csv"

# Load original data
df = pd.read_excel(RAW_PATH)

print("Original rows:", len(df))

# Remove exact duplicate records
df = df.drop_duplicates()
print("After removing duplicates:", len(df))

# Standardise text fields
df["Description"] = df["Description"].astype("string").str.strip()
df["Country"] = df["Country"].astype("string").str.strip()

# Remove records without customer IDs or product descriptions
df = df.dropna(subset=["CustomerID", "Description"])

# Exclude invalid quantities and prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Standardise customer IDs
df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)

# Calculate transaction value
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Sort by transaction date
df = df.sort_values("InvoiceDate").reset_index(drop=True)

# Save cleaned dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned rows:", len(df))
print("Unique customers:", df["CustomerID"].nunique())
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
print("Saved to:", OUTPUT_PATH)
