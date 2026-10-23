"""Head B online engine: antecedent-indexed association rules + popularity fallback.

Mining is offline (src/build_head_b_rules.py). Rules are loaded once at startup.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


class RulesService:
    def __init__(self, rules_path: Path, popularity_path: Path, products_path: Path):
        with open(rules_path, encoding="utf-8") as fh:
            artifact = json.load(fh)
        self.header: dict = artifact["header"]
        self.rules: List[dict] = artifact["rules"]
        self.index: Dict[str, List[dict]] = defaultdict(list)
        for r in self.rules:
            self.index[r["antecedent"]].append(r)
        for ant in self.index:
            self.index[ant].sort(key=lambda r: r["lift"], reverse=True)
        with open(popularity_path, encoding="utf-8") as fh:
            self.popularity: List[str] = json.load(fh)["ranking"]
        with open(products_path, encoding="utf-8") as fh:
            self.products: Dict[str, dict] = json.load(fh)

    def describe(self, code: Optional[str]) -> Optional[str]:
        if code is None:
            return None
        p = self.products.get(code)
        return p["description"] if p else None

    def recommend(self, basket: Sequence[str], k: int = 5) -> List[dict]:
        basket_set = {b.strip() for b in basket if b and b.strip()}
        if not basket_set:
            return self._popular(basket_set, k)

        best: Dict[str, dict] = {}
        for item in basket_set:
            for rule in self.index.get(item, []):
                cons = rule["consequent"]
                if cons in basket_set:
                    continue
                if cons not in best or rule["lift"] > best[cons]["lift"]:
                    best[cons] = {**rule, "because_of": item}

        ranked = sorted(best.values(), key=lambda r: (r["lift"], r["confidence"]), reverse=True)[:k]
        recs = [
            {
                "item": r["consequent"], "description": self.describe(r["consequent"]),
                "confidence": r["confidence"], "lift": r["lift"],
                "because_of": r["because_of"], "because_of_description": self.describe(r["because_of"]),
                "strategy": "rule",
            }
            for r in ranked
        ]
        if len(recs) < k:
            taken = basket_set | {r["item"] for r in recs}
            recs += self._popular(taken, k - len(recs))
        return recs

    def _popular(self, exclude: set, k: int) -> List[dict]:
        out: List[dict] = []
        for code in self.popularity:
            if code in exclude:
                continue
            out.append({"item": code, "description": self.describe(code), "strategy": "popularity"})
            if len(out) == k:
                break
        return out
