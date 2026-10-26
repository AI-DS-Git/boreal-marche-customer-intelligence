"""Latency profile for the retention model and the API (D5 performance section).

In-process mode (default): model load, first (cold) vs warm prediction, the effect of loading at start-up versus
per request, n_jobs=-1 (as saved) vs n_jobs=1, batch scaling, and explanation cost.
Live mode:  python scripts/perf_profile.py --api https://<api-host> [--n 60]
            times each endpoint over the network (p50/p95/p99, ms), sequentially, one client.
Writes reports/performance/{inprocess.csv,live_api.csv,summary.json}. Standard library + the project only.
"""
from __future__ import annotations
import argparse
import json
import statistics as st
import sys
import time
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, ".")
OUT = Path("reports/performance")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summ(xs):
    return dict(n=len(xs), p50=round(st.median(xs), 2), p95=round(pct(xs, .95), 2), p99=round(pct(xs, .99), 2), mean=round(st.mean(xs), 2))


def timeit(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def inprocess(n: int) -> list[dict]:
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    import joblib  # noqa: F401
    from app.config import get_settings
    from app.services.model_service import ModelService
    from tests.helpers import golden_features
    imports_ms = (time.perf_counter() - t0) * 1000
    s = get_settings()
    f = golden_features()
    rows = []
    t = time.perf_counter()
    m = ModelService(s.model_path, s.model_metadata_path, False, reference_path=s.drift_reference_path)
    load_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    m.predict_proba([f])
    first_ms = (time.perf_counter() - t) * 1000
    rows += [dict(case="imports (joblib, sklearn, app)", ms=round(imports_ms, 1)),
             dict(case="model load at start-up (once)", ms=round(load_ms, 1)),
             dict(case="first prediction after load (cold)", ms=round(first_ms, 1))]
    warm = timeit(lambda: m.predict_proba([f]), n)
    rows.append(dict(case="warm single prediction (n_jobs=1, as served)", **summ(warm)))
    # BEFORE: what a per-request model load would cost
    per_req = timeit(lambda: ModelService(s.model_path, s.model_metadata_path, False).predict_proba([f]), max(5, n // 6))
    rows.append(dict(case="load model on EVERY request (rejected design)", **summ(per_req)))
    # BEFORE: forest as saved (n_jobs=-1)
    est = m._raw_model().named_steps["model"]
    est.n_jobs = -1
    rows.append(dict(case="warm single prediction (n_jobs=-1, as saved)", **summ(timeit(lambda: m.predict_proba([f]), n))))
    for b in (100, 1000):
        items = [f] * b
        rows.append(dict(case=f"batch {b} (n_jobs=-1), total ms", **summ(timeit(lambda items=items: m.predict_proba(items), 10))))
    est.n_jobs = 1
    for b in (100, 1000):
        items = [f] * b
        rows.append(dict(case=f"batch {b} (n_jobs=1), total ms", **summ(timeit(lambda items=items: m.predict_proba(items), 10))))
    rows.append(dict(case="/explain work (group occlusion, 1 batched call)", **summ(timeit(lambda: m.explain(f, 3), n))))
    # naive explanation: one predict call per group
    def naive():
        for _ in m.GROUPS:
            m.predict_proba([f])
    rows.append(dict(case="explain done naively (one call per group)", **summ(timeit(naive, max(5, n // 6)))))
    return rows


def http(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return (time.perf_counter() - t) * 1000


def live(base: str, n: int) -> list[dict]:
    from tests.helpers import golden
    g = golden()["features"]
    cases = [("GET /health", "GET", "/health", None), ("GET /ready", "GET", "/ready", None), ("GET /version", "GET", "/version", None),
             ("POST /predict", "POST", "/predict", {"features": g}), ("POST /explain", "POST", "/explain", {"features": g, "k": 3}),
             ("POST /predict/batch (100)", "POST", "/predict/batch", {"items": [g] * 100}),
             ("POST /recommend", "POST", "/recommend", {"basket": ["85123A"], "k": 3}),
             ("POST /chat (rule provider)", "POST", "/chat", {"message": "What is your return policy?"})]
    rows = []
    for name, meth, path, body in cases:
        try:
            http(base, meth, path, body)  # first call, not counted
            rows.append(dict(case=name, **summ([http(base, meth, path, body) for _ in range(n)])))
        except Exception as e:  # keep going: a missing endpoint is itself a finding
            rows.append(dict(case=name, error=str(e)[:80]))
        print(rows[-1])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    if a.api:
        r = live(a.api.rstrip("/"), a.n)
        pd.DataFrame(r).to_csv(OUT / "live_api.csv", index=False)
    else:
        r = inprocess(a.n)
        pd.DataFrame(r).to_csv(OUT / "inprocess.csv", index=False)
        for x in r:
            print(x)


if __name__ == "__main__":
    main()
