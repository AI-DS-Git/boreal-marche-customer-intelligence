import json

import numpy as np
import pandas as pd

from app.config import get_settings
from app.db import LogStore
from app.monitoring import MonitoringService, build_reference, psi


def test_psi_zero_for_identical_and_large_for_shifted():
    a = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(a, a) < 1e-9
    assert psi(a, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def _store_with(values, n_threshold=40):
    store = LogStore("sqlite:///:memory:")
    for i, v in enumerate(values):
        store.log_prediction(f"r{i}", "/predict", "v", "sha", {"x": float(v)}, 0.5, "no_action", 5.0)
        store.log_request(f"r{i}", "/predict", 200, 5.0)
    return store


def test_drift_detected_only_when_distribution_shifts():
    rng = np.random.default_rng(0)
    ref = build_reference(pd.DataFrame({"x": rng.normal(0, 1, 2000)}), rng.uniform(0, 1, 2000))
    s = get_settings()

    same = MonitoringService(_store_with(rng.normal(0, 1, 200)), ref, s).summary()
    shifted = MonitoringService(_store_with(rng.normal(3, 1, 200)), ref, s).summary()
    assert same["drift"]["features"][0]["drift_status"] == "green"
    assert shifted["drift"]["features"][0]["drift_status"] == "red"


def test_alert_raised_once_per_cooldown():
    rng = np.random.default_rng(0)
    ref = build_reference(pd.DataFrame({"x": rng.normal(0, 1, 2000)}), rng.uniform(0, 1, 2000))
    mon = MonitoringService(_store_with(rng.normal(3, 1, 200)), ref, get_settings())
    summary = mon.summary()
    assert mon.raise_alerts(summary)        # first time: raised
    assert not mon.raise_alerts(summary)    # same condition inside cooldown: suppressed


def test_too_little_traffic_reports_insufficient_data():
    mon = MonitoringService(_store_with([0.1] * 5), {"features": {}, "prediction": {}}, get_settings())
    assert mon.summary()["drift"]["status"] == "insufficient_data"


def test_delayed_ground_truth_is_explicit():
    store = _store_with([0.1] * 3)
    gt = MonitoringService(store, None, get_settings()).summary()["ground_truth"]
    assert gt["accuracy"] is None and gt["first_true_accuracy_signal_on"]
    store.record_outcome("r0", 1)
    gt = MonitoringService(store, None, get_settings()).summary()["ground_truth"]
    assert gt["accuracy"]["n_labelled"] == 1
