"""Drift, data-quality and operational monitoring (spec 3.6).

Input drift: PSI and Kolmogorov-Smirnov per numeric feature against a training
reference. Prediction drift: PSI on the predicted probability. Data-quality drift: share
of values outside the training range, and change in null rate. Operational: request
volume, latency percentiles, error rate, recommendation rule fire-rate. Each metric has
an explicit threshold and a traffic-light status; breaches raise an alert.

Delayed ground truth: the retention label is only observable 90 days after a prediction.
Outcomes are written to the `outcomes` table when they arrive (backfill); until then
accuracy is reported as unavailable and the date of the first possible true accuracy
signal is shown.
"""
import json
import math
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
LABEL_DELAY_DAYS = 90
SAMPLE_SIZE = 500
GREEN, AMBER, RED, NO_DATA = "green", "amber", "red", "insufficient_data"
_RANK = {GREEN: 0, NO_DATA: 0, AMBER: 1, RED: 2}


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_noise_floor(n_bins: int, n_ref: int, n_cur: int) -> float:
    """Expected PSI between two samples of the SAME distribution: about (k-1)(1/n_ref + 1/n_cur).
    Small monitoring windows would otherwise false-alarm on pure sampling noise."""
    return (n_bins - 1) * (1.0 / max(n_ref, 1) + 1.0 / max(n_cur, 1))


def _bin_props(values: np.ndarray, edges: List[float]) -> np.ndarray:
    idx = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def numeric_reference(values: pd.Series, bins: int = 10) -> Dict[str, Any]:
    v = values.dropna().astype(float).to_numpy()
    edges = sorted(set(np.quantile(v, np.linspace(0, 1, bins + 1)).tolist()))
    if len(edges) < 3:  # near-constant feature: a single wide bin
        edges = [float(v.min()) - 1.0, float(v.max()) + 1.0]
    edges[0], edges[-1] = -math.inf, math.inf
    json_edges = [None if math.isinf(e) else e for e in edges]
    rng = np.random.default_rng(42)
    sample = rng.choice(v, size=min(SAMPLE_SIZE, len(v)), replace=False)
    return {
        "edges": json_edges,
        "props": _bin_props(v, edges).tolist(),
        "min": float(v.min()), "max": float(v.max()),
        "null_rate": float(values.isna().mean()),
        "sample": [float(x) for x in sample],
    }


def build_reference(features: pd.DataFrame, probabilities: np.ndarray) -> Dict[str, Any]:
    ref = {"n": int(len(features)), "features": {}, "prediction": numeric_reference(pd.Series(probabilities))}
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            ref["features"][col] = numeric_reference(features[col])
    return ref


def _edges(entry: Dict[str, Any]) -> List[float]:
    e = entry["edges"]
    return [-math.inf if x is None and i == 0 else (math.inf if x is None else x) for i, x in enumerate(e)]


def _status(value: float, warn: float, alert: float) -> str:
    return RED if value >= alert else AMBER if value >= warn else GREEN


def drift_report(reference: Dict[str, Any], features: pd.DataFrame, probabilities: np.ndarray,
                 psi_warn: float, psi_alert: float) -> Dict[str, Any]:
    rows = []
    for name, entry in reference["features"].items():
        if name not in features.columns:
            continue
        cur = features[name].dropna().astype(float).to_numpy()
        if len(cur) == 0:
            continue
        raw = psi(np.array(entry["props"]), _bin_props(cur, _edges(entry)))
        p = max(0.0, raw - psi_noise_floor(len(entry["props"]), reference["n"], len(cur)))
        ks = stats.ks_2samp(np.array(entry["sample"]), cur)
        out_of_range = float(np.mean((cur < entry["min"]) | (cur > entry["max"])))
        null_delta = float(features[name].isna().mean() - entry["null_rate"])
        quality = RED if out_of_range > 0.20 or null_delta > 0.20 else AMBER if (out_of_range > 0.05 or null_delta > 0.10) else GREEN
        rows.append({
            "feature": name, "psi": round(p, 4), "psi_raw": round(raw, 4), "ks_statistic": round(float(ks.statistic), 4),
            "ks_pvalue": round(float(ks.pvalue), 6), "drift_status": _status(p, psi_warn, psi_alert),
            "out_of_range_share": round(out_of_range, 4), "null_rate_delta": round(null_delta, 4),
            "quality_status": quality,
        })
    pred_psi = None
    pred_status = NO_DATA
    if len(probabilities):
        pe = reference["prediction"]
        pred_psi = max(0.0, psi(np.array(pe["props"]), _bin_props(np.asarray(probabilities, dtype=float), _edges(pe)))
                       - psi_noise_floor(len(pe["props"]), reference["n"], len(probabilities)))
        pred_status = _status(pred_psi, psi_warn, psi_alert)
    return {"features": rows, "prediction_psi": None if pred_psi is None else round(pred_psi, 4),
            "prediction_status": pred_status}


def percentile(values: List[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def ops_report(requests: List[Dict[str, Any]], settings) -> Dict[str, Any]:
    total = len(requests)
    lat = [r["latency_ms"] for r in requests]
    errors = sum(1 for r in requests if r["status_code"] >= 500)
    reco = [r for r in requests if r["endpoint"] == "/recommend"]
    fired = 0
    for r in reco:
        meta = json.loads(r["meta_json"]) if r.get("meta_json") else {}
        if meta.get("rule_hits", 0) > 0:
            fired += 1
    err_rate = errors / total if total else 0.0
    p95 = percentile(lat, 95)
    return {
        "request_count": total,
        "error_rate": round(err_rate, 4),
        "error_status": RED if err_rate >= settings.error_rate_alert else GREEN if total else NO_DATA,
        "latency_ms": {"p50": percentile(lat, 50), "p95": p95, "p99": percentile(lat, 99)},
        "latency_status": (RED if p95 is not None and p95 >= settings.p95_latency_ms_alert else GREEN) if total else NO_DATA,
        "recommend_requests": len(reco),
        "rule_fire_rate": round(fired / len(reco), 4) if reco else None,
    }


class MonitoringService:
    def __init__(self, store, reference: Optional[Dict[str, Any]], settings):
        self.store, self.reference, self.settings = store, reference, settings

    def _features_frame(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        recs = [json.loads(r["features_json"]) for r in rows]
        df = pd.DataFrame(recs)
        # Same encoding as training: dates -> days since 2009-12-01.
        epoch = pd.Timestamp("2009-12-01")
        for c in ("first_purchase_date", "last_purchase_date"):
            if c in df:
                df[c + "_days"] = (pd.to_datetime(df[c]) - epoch).dt.days
        return df

    def summary(self, hours: float = 24) -> Dict[str, Any]:
        s = self.settings
        preds = self.store.recent_predictions(hours)
        reqs = self.store.recent_requests(hours)
        ops = ops_report(reqs, s)
        drift: Dict[str, Any] = {"status": NO_DATA, "n_predictions": len(preds), "features": [], "prediction_psi": None}
        if self.reference is not None and len(preds) >= s.min_requests_for_drift:
            df = self._features_frame(preds)
            probs = np.array([p["probability"] for p in preds])
            rep = drift_report(self.reference, df, probs, s.psi_warn, s.psi_alert)
            worst = max([r["drift_status"] for r in rep["features"]] + [rep["prediction_status"]],
                        key=lambda x: _RANK[x], default=NO_DATA)
            quality = max([r["quality_status"] for r in rep["features"]], key=lambda x: _RANK[x], default=NO_DATA)
            drift.update({"status": worst, "quality_status": quality, **rep})
        first = self.store.first_prediction_ts()
        pairs = self.store.labelled_pairs()
        accuracy = None
        if pairs:
            ok = sum(1 for p in pairs if (p["probability"] >= 0.5) == bool(p["observed_retained"]))
            accuracy = {"n_labelled": len(pairs), "accuracy_at_0.5": round(ok / len(pairs), 4)}
        first_signal = None
        if first:
            first_signal = (datetime.fromisoformat(first) + timedelta(days=LABEL_DELAY_DAYS)).date().isoformat()
        ground_truth = {
            "label_delay_days": LABEL_DELAY_DAYS, "first_true_accuracy_signal_on": first_signal,
            "accuracy": accuracy,
            "note": "Retention is only observable 90 days after a prediction; labels are backfilled into the outcomes table.",
        }
        statuses = [drift["status"], drift.get("quality_status", NO_DATA), ops["error_status"], ops["latency_status"]]
        overall = max(statuses, key=lambda x: _RANK[x])
        return {"window_hours": hours, "overall_status": overall, "drift": drift, "ops": ops,
                "ground_truth": ground_truth, "alerts": self.store.recent_alerts(10)}

    def raise_alerts(self, summary: Dict[str, Any], cooldown_hours: float = 6) -> List[Dict[str, str]]:
        """Write (and optionally POST) one alert per red condition, at most once per cooldown."""
        found: List[Dict[str, str]] = []
        d, o = summary["drift"], summary["ops"]
        for r in d.get("features", []):
            if r["drift_status"] == RED:
                found.append({"key": f"drift:{r['feature']}", "severity": "red",
                              "message": f"Input drift on {r['feature']}: PSI {r['psi']} >= {self.settings.psi_alert}"})
        if d.get("prediction_status") == RED:
            found.append({"key": "drift:prediction", "severity": "red",
                          "message": f"Prediction drift: PSI {d['prediction_psi']} >= {self.settings.psi_alert}"})
        if o["error_status"] == RED:
            found.append({"key": "ops:errors", "severity": "red", "message": f"Error rate {o['error_rate']:.1%}"})
        if o["latency_status"] == RED:
            found.append({"key": "ops:latency", "severity": "red", "message": f"p95 latency {o['latency_ms']['p95']:.0f} ms"})
        sent = []
        for a in found:
            if self.store.alert_exists_since(a["key"], cooldown_hours):
                continue
            self.store.log_alert(a["key"], a["severity"], a["message"])
            sent.append(a)
            self._notify(a)
        return sent

    def _notify(self, alert: Dict[str, str]) -> None:
        url = self.settings.alert_webhook_url
        if not url:
            return
        body = json.dumps({"text": f"[{alert['severity'].upper()}] {alert['message']}"}).encode()
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)  # noqa: S310 (operator-configured URL)
        except Exception:
            pass  # an unreachable webhook must never break the API
