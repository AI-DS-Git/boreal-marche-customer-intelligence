import pandas as pd
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

INPUT = "data/processed/silver_transactions.csv"
RULES_OUTPUT = "docs/head_b_association_rules.csv"
ITEMSETS_OUTPUT = "docs/head_b_frequent_itemsets.csv"

print("=" * 80)
print("HEAD B - ASSOCIATION RULE ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    INPUT,
    usecols=["Invoice", "StockCode", "Customer ID"],
    low_memory=False
)

df = df[
    df["Customer ID"].notna()
    & df["StockCode"].notna()
].copy()

# One product is represented once per invoice.
basket = (
    df.drop_duplicates(["Invoice", "StockCode"])
      .assign(purchased=1)
      .pivot_table(
          index="Invoice",
          columns="StockCode",
          values="purchased",
          fill_value=0
      )
)

basket = basket.astype(bool)

print(f"Customer-attributed rows: {len(df):,}")
print(f"Unique baskets: {basket.shape[0]:,}")
print(f"Unique products: {basket.shape[1]:,}")

# Initial conservative support threshold.
MIN_SUPPORT = 0.01

print()
print(f"Minimum support: {MIN_SUPPORT:.2%}")
print("Mining frequent itemsets...")

frequent_itemsets = apriori(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=2
)

frequent_itemsets["itemset_size"] = frequent_itemsets["itemsets"].apply(len)

print(f"Frequent itemsets found: {len(frequent_itemsets):,}")

# Generate rules from the frequent itemsets.
if len(frequent_itemsets) > 0:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.20
    )

    if len(rules) > 0:
        rules = rules[
            [
                "antecedents",
                "consequents",
                "antecedent support",
                "consequent support",
                "support",
                "confidence",
                "lift"
            ]
        ].copy()

        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(map(str, x)))
        )
        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(map(str, x)))
        )

        rules = rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=False
        )

        rules.to_csv(RULES_OUTPUT, index=False)

        print(f"Association rules found: {len(rules):,}")
        print()
        print("Top 10 rules:")
        print(rules.head(10).to_string(index=False))
        print()
        print(f"Saved: {RULES_OUTPUT}")
    else:
        print("No association rules met the confidence threshold.")
else:
    print("No frequent itemsets found.")

frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
    lambda x: ", ".join(sorted(map(str, x)))
)

frequent_itemsets = frequent_itemsets.sort_values(
    "support",
    ascending=False
)

frequent_itemsets.to_csv(
    ITEMSETS_OUTPUT,
    index=False
)

print(f"Saved: {ITEMSETS_OUTPUT}")
