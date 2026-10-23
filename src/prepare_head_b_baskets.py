import pandas as pd
from pathlib import Path

INPUT = "data/processed/silver_transactions.csv"
OUTPUT = "docs/head_b_association_rules.csv"

print("=" * 80)
print("HEAD B - ASSOCIATION RULE ANALYSIS")
print("=" * 80)

df = pd.read_csv(INPUT)

print(f"Silver transaction rows: {len(df):,}")

# Keep records that can be attributed to a customer and product.
df = df[
    df["Customer ID"].notna()
    & df["StockCode"].notna()
].copy()

# One product appears once per invoice for basket purposes.
basket = (
    df.groupby(["Invoice", "StockCode"])
      .size()
      .unstack(fill_value=0)
)

basket = basket.gt(0).astype("uint8")

print(f"Customer-attributed rows: {len(df):,}")
print(f"Unique baskets: {basket.shape[0]:,}")
print(f"Unique products: {basket.shape[1]:,}")

# Save basket dimensions for D2 documentation.
Path("docs").mkdir(exist_ok=True)

summary = pd.DataFrame({
    "metric": [
        "silver_rows",
        "customer_attributed_rows",
        "unique_baskets",
        "unique_products"
    ],
    "value": [
        len(pd.read_csv(INPUT)),
        len(df),
        basket.shape[0],
        basket.shape[1]
    ]
})

summary.to_csv("docs/head_b_basket_summary.csv", index=False)

print()
print("Saved: docs/head_b_basket_summary.csv")
