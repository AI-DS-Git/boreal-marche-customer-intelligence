"""Expected-profit sensitivity of the retention-offer policy to the three planning assumptions.

Reproduces the D3 profit model (src/calibrate_and_threshold.py) on the held-out slice with the DEPLOYED model and the
frozen threshold, then varies coupon cost, gross margin and recovery rate one at a time and together.
Targeted = P(retain) < threshold. A targeted customer who would have churned yields margin x recovery; every targeted
customer costs one coupon. The blanket policy targets everyone. Annual figure = per-customer gain x customers x campaigns.
"""
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from retrain import CURRENT_ART, DATA, ID, THRESHOLD, prepare, split  # noqa: E402

CUSTOMERS, CAMPAIGNS = 5249, 4
BASE = dict(coupon=5.0, margin=0.20, recovery=0.30)


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = prepare(df)
    _, (Xca, yca), (Xte, yte) = split(X, y, df[ID])
    m = joblib.load(CURRENT_ART)
    p = m.predict_proba(Xte)[:, 1]
    rev = Xte["revenue_per_order"].to_numpy()
    churn = (yte.to_numpy() == 0)
    targeted = p < THRESHOLD

    def gain(coupon, margin, recovery):
        def profit(mask):
            return (rev[mask & churn] * margin * recovery).sum() - mask.sum() * coupon
        t, b = profit(targeted), profit(np.ones_like(targeted))
        return t, b, (t - b) / len(p)

    t, b, g = gain(**BASE)
    print(f"held-out n={len(p)}  targeted={targeted.mean():.1%}  threshold={THRESHOLD}")
    print(f"BASE: targeted profit ${t:,.2f}  blanket ${b:,.2f}  gain per customer per campaign ${g:.3f}"
          f"  -> annual ${g * CUSTOMERS * CAMPAIGNS:,.0f}")
    rows = []
    for name, grid in [("coupon", [3, 4, 5, 6, 8]), ("margin", [.15, .20, .25]), ("recovery", [.10, .20, .30, .45])]:
        for v in grid:
            kw = {**BASE, name: v}
            t, b, g = gain(**kw)
            rows.append((name, v, round(t, 2), round(b, 2), round(g, 3), round(g * CUSTOMERS * CAMPAIGNS)))
    out = pd.DataFrame(rows, columns=["varied", "value", "targeted_profit", "blanket_profit", "gain_per_cust", "annual_gain"])
    print(out.to_string(index=False))
    allc = [(c, m_, r, gain(c, m_, r)[2] * CUSTOMERS * CAMPAIGNS)
            for c, m_, r in itertools.product([3, 5, 8], [.15, .20, .25], [.10, .30, .45])]
    a = pd.DataFrame(allc, columns=["coupon", "margin", "recovery", "annual_gain"])
    print(f"all 27 combinations: min ${a.annual_gain.min():,.0f}  max ${a.annual_gain.max():,.0f}; "
          f"negative in {(a.annual_gain < 0).sum()} of 27")
    print(a.sort_values("annual_gain").head(5).to_string(index=False))

    # Threshold-selection check: D3 swept the threshold on the held-out slice. Choose it on the calibration slice instead.
    grid = np.linspace(0.05, 0.95, 37)

    def profit_at(Xs, ys, thr):
        ps = m.predict_proba(Xs)[:, 1]
        rv = Xs["revenue_per_order"].to_numpy()
        tg = ps < thr
        return (rv[tg & (ys.to_numpy() == 0)] * BASE["margin"] * BASE["recovery"]).sum() - tg.sum() * BASE["coupon"]

    t_cal = grid[int(np.argmax([profit_at(Xca, yca, t) for t in grid]))]
    print(f"threshold chosen on calibration slice: {t_cal:.3f}; held-out profit there ${profit_at(Xte, yte, t_cal):,.2f}"
          f" vs ${profit_at(Xte, yte, THRESHOLD):,.2f} at {THRESHOLD}")
    Path("reports/performance").mkdir(parents=True, exist_ok=True)
    out.to_csv("reports/performance/profit_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
