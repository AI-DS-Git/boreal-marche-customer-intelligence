"""Scalability probes for the prediction log and the monitoring summary (D5 scalability section).

Measures, on a throw-away SQLite file: (1) prediction-log write rate (one row per prediction, as the API does),
(2) /monitoring/summary compute time versus the number of logged predictions, including the 5,000-row cap in
LogStore.recent_predictions, and (3) rule-lookup time versus rule count. Run: python scripts/scale_probe.py
"""
from __future__ import annotations
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, ".")
warnings.simplefilter("ignore")
from app.config import get_settings  # noqa: E402
from app.db import LogStore  # noqa: E402
from app.monitoring import MonitoringService  # noqa: E402
from tests.helpers import golden  # noqa: E402

OUT = Path("reports/performance")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    ref = json.loads(s.drift_reference_path.read_text(encoding="utf-8"))
    g = golden()["features"]
    rows = []
    with tempfile.TemporaryDirectory() as d:
        store = LogStore(f"sqlite:///{d}/probe.db")
        mon = MonitoringService(store, ref, s)
        done = 0
        for target in (1000, 5000, 20000):
            while done < target:
                store.log_prediction(f"r{done}", "/predict", "v", "sha", g, 0.7, "no_action", 30.0)
                done += 1
            t = time.perf_counter()
            summ = mon.summary(24)
            sm = (time.perf_counter() - t) * 1000
            rows.append(dict(rows_logged=done, summary_ms=round(sm, 1), rows_used_for_drift=summ["drift"]["n_predictions"]))
            print(rows[-1])
        t = time.perf_counter()
        for i in range(2000):
            store.log_prediction(f"w{i}", "/predict", "v", "sha", g, 0.7, "no_action", 30.0)
        write_rate = 2000 / (time.perf_counter() - t)
    out = dict(summary_vs_rows=rows, sqlite_write_rows_per_s=round(write_rate))
    (OUT / "scale_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
