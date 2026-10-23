"""Build models/drift_reference.json: per-feature training distribution for drift checks.

Run from the repo root:  python scripts/build_reference_stats.py
Uses the Gold table (training population) and the shipped model to get the reference
prediction distribution. No customer identifiers are stored.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings  # noqa: E402
from app.monitoring import build_reference  # noqa: E402
from app.services.model_service import EPOCH, ModelService  # noqa: E402

s = get_settings()
gold = pd.read_csv("data/processed/gold_customer_features.csv")
X = gold.drop(columns=["retained_90d", "Customer ID"]).copy()
for c in ("first_purchase_date", "last_purchase_date"):
    X[c] = pd.to_datetime(X[c])
    X[c + "_days"] = (X[c] - pd.Timestamp(EPOCH)).dt.days
    X = X.drop(columns=[c])

svc = ModelService(s.model_path, s.model_metadata_path)
X = X[svc.feature_order]
probs = svc.model.predict_proba(X)[:, 1]
ref = build_reference(X, np.asarray(probs))
Path(s.drift_reference_path).write_text(json.dumps(ref, separators=(",", ":")), encoding="utf-8")
print(f"wrote {s.drift_reference_path} ({ref['n']} customers, {len(ref['features'])} features)")
