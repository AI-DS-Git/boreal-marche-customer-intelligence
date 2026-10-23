"""
D3 — Head B online serving engine.

Mining (build_head_b_rules.py) is offline; this module is the online path
that a FastAPI /recommend endpoint would call per request (spec 3.3 / D2
Head B architecture). Design:
  - Rules are indexed by antecedent for O(1) lookup rather than scanning a
    flat rule list per request.
  - For a given basket, candidate consequents from all firing rules are
    deduplicated by consequent, keeping the highest-lift rule per item.
  - Items already in the basket are excluded from recommendations.
  - The firing rule is returned alongside each recommendation so the UI can
    explain "because you added X" (spec 3.6 front-end requirement).
  - Empty basket and no-match basket (no rule fires) both fall back to a
    global popularity ranking, computed once at load time.
"""

import json
from collections import defaultdict
from pathlib import Path


class RuleEngine:
    def __init__(self, rules_path="docs/head_b_rules.json",
                 popularity_path="docs/head_b_popularity.json"):
        with open(rules_path) as f:
            artifact = json.load(f)
        self.header = artifact["header"]
        self.rules = artifact["rules"]

        # Index by antecedent -> list of rule dicts, sorted by lift desc.
        self.index = defaultdict(list)
        for r in self.rules:
            self.index[r["antecedent"]].append(r)
        for ant in self.index:
            self.index[ant].sort(key=lambda r: r["lift"], reverse=True)

        with open(popularity_path) as f:
            self.popularity = json.load(f)["ranking"]  # list of stock codes, most popular first

    def recommend(self, basket_items, k=5):
        """Return up to k recommendations for the given set of basket items.

        Each recommendation is {"item": ..., "confidence": ..., "lift": ...,
        "because_of": ..., "strategy": "rule"|"popularity"}.
        """
        basket_set = set(basket_items)

        if not basket_set:
            return self._popularity_fallback(basket_set, k, reason="empty_basket")

        candidates = {}  # consequent -> best rule seen so far
        for item in basket_set:
            for rule in self.index.get(item, []):
                cons = rule["consequent"]
                if cons in basket_set:
                    continue
                if cons not in candidates or rule["lift"] > candidates[cons]["lift"]:
                    candidates[cons] = {**rule, "because_of": item}

        if not candidates:
            return self._popularity_fallback(basket_set, k, reason="no_match_basket")

        ranked = sorted(candidates.values(), key=lambda r: (r["lift"], r["confidence"]), reverse=True)
        recs = [{
            "item": r["consequent"], "confidence": r["confidence"], "lift": r["lift"],
            "because_of": r["because_of"], "strategy": "rule",
        } for r in ranked[:k]]

        if len(recs) < k:
            recs += self._popularity_fallback(
                basket_set | {r["item"] for r in recs}, k - len(recs), reason="partial_fill"
            )
        return recs

    def _popularity_fallback(self, exclude_items, k, reason):
        out = []
        for item in self.popularity:
            if item in exclude_items:
                continue
            out.append({"item": item, "confidence": None, "lift": None,
                        "because_of": None, "strategy": f"popularity ({reason})"})
            if len(out) >= k:
                break
        return out


def build_popularity_artifact(baskets, out_path="docs/head_b_popularity.json"):
    from collections import Counter
    counts = Counter()
    for b in baskets:
        counts.update(b)
    ranking = [item for item, _ in counts.most_common()]
    Path("docs").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"ranking": ranking}, f)
    return ranking


if __name__ == "__main__":
    import pandas as pd

    CUTOFF = pd.Timestamp("2011-08-31 23:59:59")
    silver = pd.read_csv("data/processed/silver_transactions.csv", parse_dates=["InvoiceDate"], low_memory=False)
    silver = silver[silver["Customer ID"].notna()].copy()
    train_rows = silver[silver["InvoiceDate"] <= CUTOFF]
    baskets_train = train_rows.groupby("Invoice")["StockCode"].apply(lambda s: frozenset(s.dropna()))
    baskets_train = [b for b in baskets_train.values if len(b) > 0]

    build_popularity_artifact(baskets_train)
    engine = RuleEngine()

    print("Smoke test — empty basket:")
    print(engine.recommend([], k=5))

    print("\nSmoke test — basket containing '22746' (a strong antecedent, see docs/head_b_rules.json):")
    print(engine.recommend(["22746"], k=5))

    print("\nSmoke test — basket with a popular item that has no rule as antecedent (85123A):")
    print("(Expected to fall back to popularity: 85123A's high base rate means confidence(85123A -> X)")
    print(" rarely clears the threshold, even though X -> 85123A rules are common. This is a real,")
    print(" documented asymmetry, not a bug — see D3 report.)")
    print(engine.recommend(["85123A"], k=5))

    print("\nSmoke test — basket with a nonsense/no-match item:")
    print(engine.recommend(["__NOT_A_REAL_PRODUCT__"], k=5))
