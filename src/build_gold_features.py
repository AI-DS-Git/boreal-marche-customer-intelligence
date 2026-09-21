import pandas as pd
import numpy as np
from pathlib import Path

FILE = r"data\raw\online_retail_II.xlsx"
CUTOFF = pd.Timestamp("2011-08-31 23:59:59")

# Load both UCI sheets
df1 = pd.read_excel(FILE, sheet_name="Year 2009-2010")
df2 = pd.read_excel(FILE, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)

# Standardize fields
df["Invoice"] = df["Invoice"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Apply established Silver cleaning rules
df = df.drop_duplicates()
df = df[~df["Invoice"].str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df = df[df["InvoiceDate"].notna()]

# Observation period only
df = df[df["InvoiceDate"] <= CUTOFF].copy()

# Customer-level population
df = df[df["Customer ID"].notna()].copy()

# Transaction value
df["Revenue"] = df["Quantity"] * df["Price"]

# Sort for time-based calculations
df = df.sort_values(["Customer ID", "InvoiceDate"])

# ------------------------------------------------------------
# Customer-level features
# ------------------------------------------------------------

reference_date = CUTOFF

features = df.groupby("Customer ID").agg(
    recency_days=("InvoiceDate",
                  lambda x: (reference_date - x.max()).days),

    frequency_orders=("Invoice",
                       "nunique"),

    monetary_total=("Revenue",
                    "sum"),

    monetary_mean_order=("Revenue",
                         lambda x: x.sum() / df.loc[x.index, "Invoice"].nunique()),

    average_quantity=("Quantity",
                      "mean"),

    total_quantity=("Quantity",
                    "sum"),

    unique_products=("StockCode",
                     "nunique"),

    unique_product_descriptions=("Description",
                                 "nunique"),

    average_unit_price=("Price",
                        "mean"),

    country=("Country",
             "first"),

    first_purchase_date=("InvoiceDate",
                         "min"),

    last_purchase_date=("InvoiceDate",
                        "max"),

    active_days=("InvoiceDate",
                 lambda x: x.dt.date.nunique()),

    active_months=("InvoiceDate",
                   lambda x: x.dt.to_period("M").nunique()),

    weekend_transactions=("InvoiceDate",
                           lambda x: (x.dt.dayofweek >= 5).sum()),

    transaction_count=("Invoice",
                       "count")
)

# Additional derived features
features["customer_lifetime_days"] = (
    features["last_purchase_date"] -
    features["first_purchase_date"]
).dt.days

features["orders_per_active_month"] = (
    features["frequency_orders"] /
    features["active_months"].replace(0, np.nan)
)

features["products_per_order"] = (
    features["unique_products"] /
    features["frequency_orders"].replace(0, np.nan)
)

features["revenue_per_order"] = (
    features["monetary_total"] /
    features["frequency_orders"].replace(0, np.nan)
)

features["quantity_per_order"] = (
    features["total_quantity"] /
    features["frequency_orders"].replace(0, np.nan)
)

features["weekend_transaction_ratio"] = (
    features["weekend_transactions"] /
    features["transaction_count"].replace(0, np.nan)
)

# Replace infinite values
features = features.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# Create 90-day retention label
# ------------------------------------------------------------

LABEL_START = CUTOFF
LABEL_END = CUTOFF + pd.Timedelta(days=90)

future = pd.concat([df1, df2], ignore_index=True)

future["Invoice"] = future["Invoice"].astype(str)
future["InvoiceDate"] = pd.to_datetime(
    future["InvoiceDate"],
    errors="coerce"
)

# Apply same transaction rules
future = future.drop_duplicates()
future = future[~future["Invoice"].str.startswith("C")]
future = future[future["Quantity"] > 0]
future = future[future["Price"] > 0]
future = future[future["InvoiceDate"].notna()]
future = future[future["Customer ID"].notna()]

future = future[
    (future["InvoiceDate"] > LABEL_START) &
    (future["InvoiceDate"] <= LABEL_END)
]

retained_customers = set(future["Customer ID"])

features["retained_90d"] = (
    features.index.to_series()
    .isin(retained_customers)
    .astype(int)
)

# ------------------------------------------------------------
# Clean output
# ------------------------------------------------------------

# Keep customer ID as a normal column
features = features.reset_index()

# Convert country to string
features["country"] = features["country"].astype(str)

# Save
Path("data/processed").mkdir(parents=True, exist_ok=True)

output = "data/processed/gold_customer_features.csv"
features.to_csv(output, index=False)

print("=" * 75)
print("GOLD CUSTOMER FEATURE DATASET")
print("=" * 75)

print(f"Observation cutoff: {CUTOFF}")
print(f"Label end:          {LABEL_END}")
print(f"Customers:          {len(features):,}")
print(f"Features + label:   {len(features.columns):,}")
print()

print("Target distribution:")
print(features["retained_90d"].value_counts().sort_index().to_string())
print()

print("Target percentage:")
print(features["retained_90d"].value_counts(normalize=True).sort_index().to_string())
print()

print("Feature columns:")
for column in features.columns:
    print(f"  - {column}")

print()
print("Missing values:")
print(
    features.isna().sum()
    .sort_values(ascending=False)
    .to_string()
)

print()
print("Saved:")
print(output)
