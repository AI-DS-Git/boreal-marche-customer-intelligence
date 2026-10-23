"""
D3 — Head B offline evaluation: temporal-holdout hit-rate@5 and catalog
coverage for the rule engine vs. the required global-popularity baseline.

Protocol:
  - Training baskets: Invoice date <= CUTOFF (used to mine rules and to rank
    popularity — see build_head_b_rules.py / head_b_serving.py).
  - Holdout baskets: Invoice date in (CUTOFF, CUTOFF + 90 days], same label
    window used for Head A, so the two heads' "future" is the same period.
  - Eligible customer: has >= 1 basket before the cutoff AND >= 1 basket in
    the holdout window. Recommendations are generated from their LAST
    pre-cutoff basket; a hit is scored if any of the top-5 recommended
    items appears anywhere in their holdout-window purchases.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

from head_b_serving import RuleEngine, build_popularity_artifact

CUTOFF = pd.Timestamp("2011-08-31 23:59:59")
LABEL_END = CUTOFF + pd.Timedelta(days=90)

silver = pd.read_csv("data/processed/silver_transactions.csv", parse_dates=["InvoiceDate"], low_memory=False)
silver = silver[silver["Customer ID"].notna()].copy()

train_rows = silver[silver["InvoiceDate"] <= CUTOFF]
holdout_rows = silver[(silver["InvoiceDate"] > CUTOFF) & (silver["InvoiceDate"] <= LABEL_END)]

train_baskets = train_rows.groupby("Invoice").agg(
    CustomerID=("Customer ID", "first"),
    InvoiceDate=("InvoiceDate", "first"),
    items=("StockCode", lambda s: frozenset(s.dropna())),
)
train_baskets = train_baskets[train_baskets["items"].apply(len) > 0]

# Last pre-cutoff basket per customer
last_basket_per_customer = (
    train_baskets.sort_values("InvoiceDate")
    .groupby("CustomerID")
    .tail(1)
    .set_index("CustomerID")["items"]
)

holdout_items_per_customer = (
    holdout_rows.groupby("Customer ID")["StockCode"]
    .apply(lambda s: frozenset(s.dropna()))
)

eligible_customers = sorted(
    set(last_basket_per_customer.index) & set(holdout_items_per_customer.index)
)
print(f"Eligible customers for temporal-holdout evaluation: {len(eligible_customers):,}")

engine = RuleEngine()
train_basket_list = [b for b in train_baskets["items"].values if len(b) > 0]
popularity_ranking = build_popularity_artifact(train_basket_list)


def popularity_recommend(exclude_items, k=5):
    out = []
    for item in popularity_ranking:
        if item in exclude_items:
            continue
        out.append(item)
        if len(out) >= k:
            break
    return out


def evaluate(recommender_fn, name, k=5):
    hits = 0
    n = 0
    recommended_union = set()
    rule_based_count = 0
    for cust in eligible_customers:
        basket = last_basket_per_customer[cust]
        actual_future = holdout_items_per_customer[cust]
        recs = recommender_fn(basket, k)
        rec_items = [r["item"] if isinstance(r, dict) else r for r in recs]
        recommended_union.update(rec_items)
        if any(r != "popularity" for r in
               ([x["strategy"] for x in recs] if recs and isinstance(recs[0], dict) else [])):
            rule_based_count += 1
        hit = len(set(rec_items) & actual_future) > 0
        hits += int(hit)
        n += 1

    hit_rate = hits / n if n else float("nan")
    catalog_size = len(set().union(*train_baskets["items"].values))
    coverage = len(recommended_union) / catalog_size
    print(f"\n[{name}] hit-rate@{k} = {hit_rate:.4f} ({hits}/{n})  |  "
          f"catalog coverage = {coverage:.4f} ({len(recommended_union)}/{catalog_size} products ever recommended)")
    if name == "Rule engine":
        print(f"  customers served by an actual rule (not pure fallback): {rule_based_count}/{n} "
              f"({rule_based_count / n:.1%})")
    return {"strategy": name, "hit_rate_at_5": hit_rate, "hits": hits, "n_customers": n,
            "catalog_coverage": coverage, "products_recommended": len(recommended_union),
            "catalog_size": catalog_size}


print("\n" + "=" * 90)
print("D3 — HEAD B TEMPORAL HOLDOUT EVALUATION")
print("=" * 90)

results = []
results.append(evaluate(lambda basket, k: engine.recommend(list(basket), k), "Rule engine"))
results.append(evaluate(lambda basket, k: [{"item": i, "strategy": "popularity"} for i in
                                            popularity_recommend(set(basket), k)], "Popularity baseline"))

results_df = pd.DataFrame(results)
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(results_df.to_string(index=False))

lift_over_baseline = results_df.loc[0, "hit_rate_at_5"] / results_df.loc[1, "hit_rate_at_5"] \
    if results_df.loc[1, "hit_rate_at_5"] > 0 else float("inf")
print(f"\nRule engine hit-rate@5 is {lift_over_baseline:.2f}x the popularity baseline.")
if lift_over_baseline < 1:
    print("HONEST FINDING: the rule engine does NOT beat the popularity baseline on hit-rate@5.")
    print("Per spec 3.3: 'If your rules do not beat popularity, report that. It is a legitimate")
    print("finding and it will be graded as such.' See D3 report Limitations for discussion.")

Path("docs").mkdir(exist_ok=True)
results_df.to_csv("docs/head_b_holdout_evaluation.csv", index=False)
print("\nSaved: docs/head_b_holdout_evaluation.csv")
