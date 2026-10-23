import pandas as pd
from pathlib import Path

RULES = "docs/head_b_association_rules.csv"
SILVER = "data/processed/silver_transactions.csv"
OUTPUT = "docs/head_b_top10_rules.csv"

rules = pd.read_csv(RULES)

products = pd.read_csv(
    SILVER,
    usecols=["StockCode", "Description"],
    low_memory=False
)

products["StockCode"] = products["StockCode"].astype(str)
products["Description"] = products["Description"].fillna("Unknown product")

product_lookup = (
    products.drop_duplicates("StockCode")
    .set_index("StockCode")["Description"]
    .to_dict()
)

def description(stock_code):
    code = str(stock_code).strip()
    return product_lookup.get(code, code)

rules["antecedent_product"] = rules["antecedents"].apply(
    lambda x: description(x.split(",")[0])
)

rules["consequent_product"] = rules["consequents"].apply(
    lambda x: description(x.split(",")[0])
)

top10 = rules[
    [
        "antecedent_product",
        "consequent_product",
        "support",
        "confidence",
        "lift"
    ]
].head(10).copy()

top10["support_pct"] = top10["support"] * 100
top10["confidence_pct"] = top10["confidence"] * 100

top10 = top10[
    [
        "antecedent_product",
        "consequent_product",
        "support_pct",
        "confidence_pct",
        "lift"
    ]
]

Path("docs").mkdir(exist_ok=True)
top10.to_csv(OUTPUT, index=False)

print("=" * 90)
print("HEAD B - TOP 10 ASSOCIATION RULES")
print("=" * 90)
print(top10.to_string(index=False))
print()
print(f"Saved: {OUTPUT}")
