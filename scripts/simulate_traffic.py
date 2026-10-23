"""Populate the monitoring dashboard with realistic and deliberately shifted traffic.

    python scripts/simulate_traffic.py --api http://localhost:8000 --n 300            # normal
    python scripts/simulate_traffic.py --api http://localhost:8000 --n 300 --drift    # shifted
    python scripts/simulate_traffic.py --api http://localhost:8000 --n 100 --errors   # bad requests

Normal traffic samples real customers from tests/golden/sample_customers.json (600
de-identified customers drawn from the training population). --drift
multiplies spend/frequency and ages the customers, which should turn the dashboard red.
Standard library only, so it runs anywhere.
"""
import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "golden" / "sample_customers.json"
GOLDEN = ROOT / "tests" / "golden" / "customer_12347.json"
INT_FIELDS = {"recency_days", "frequency_orders", "total_quantity", "unique_products",
              "unique_product_descriptions", "active_days", "active_months", "weekend_transactions",
              "transaction_count", "holiday_transactions", "customer_lifetime_days", "is_uk"}
OPTIONAL = {"price_std", "avg_days_between_orders"}


def post(base, path, body, timeout=15):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def load_population():
    """600 real, de-identified customers shipped with the repo; falls back to the Gold table.
    Refuses to continue with a single-customer population: identical rows look like extreme
    drift to the monitor, which would make 'normal' traffic turn the dashboard red."""
    if SAMPLE.exists():
        return json.loads(SAMPLE.read_text(encoding="utf-8"))["customers"]
    sys.exit(f"missing {SAMPLE}; copy tests/golden/sample_customers.json from the D4 package")


def shifted(f, rng):
    g = dict(f)
    g["frequency_orders"] = max(1, int(f["frequency_orders"] * rng.uniform(3, 6)))
    g["transaction_count"] = max(g["frequency_orders"], int(f["transaction_count"] * 4))
    g["monetary_total"] = round(f["monetary_total"] * rng.uniform(3, 6), 2)
    g["monetary_mean_order"] = round(g["monetary_total"] / g["frequency_orders"], 2)
    g["revenue_per_order"] = g["monetary_mean_order"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drift", action="store_true", help="send deliberately shifted customers")
    ap.add_argument("--errors", action="store_true", help="also send malformed requests (422s)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pop = load_population()
    codes = ["85123A", "22423", "47566", "84879", "21212"]
    counts = {}
    t0 = time.time()
    for i in range(a.n):
        f = rng.choice(pop)
        if a.drift:
            f = shifted(f, rng)
        s = post(a.api, "/predict", {"features": f})
        counts[s] = counts.get(s, 0) + 1
        if i % 3 == 0:
            post(a.api, "/recommend", {"basket": rng.sample(codes, rng.randint(0, 2)), "k": 5})
        if i % 10 == 0:
            post(a.api, "/chat", {"message": rng.choice(
                ["What is your return policy?", "Comment fonctionne la livraison ?", "Tell me a joke"])})
        if a.errors and i % 5 == 0:
            post(a.api, "/predict", {"features": {"recency_days": -1}})
    print(f"sent {a.n} predictions in {time.time()-t0:.1f}s; status counts: {counts}")
    print("Open the Monitoring page (or GET /monitoring/summary) to see the effect.")
    return 0 if counts.get(200, 0) else 1


if __name__ == "__main__":
    sys.exit(main())
