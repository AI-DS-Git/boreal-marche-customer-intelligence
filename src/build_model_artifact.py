"""
D3 — Final Head A model artifact + metadata sidecar (spec 3.5 / D3 rubric).

Ships ONE joblib file containing the complete inference pipeline
(preprocessing + calibrated estimator together), so serving code never
needs to duplicate preprocessing logic. A metadata sidecar records
everything needed to audit or roll back the artifact: model version,
training timestamp, row count, feature names/order, library versions,
git commit, decision threshold, and the held-out test metric.

Chosen model: Extra Trees (isotonic-calibrated) — ranked #1 by mean PR-AUC
in the spot-check (docs/head_a_spot_check_summary.csv) and confirmed #1 on
the held-out test (docs/final_heldout_test_results.csv). This artifact uses
the SAME hyperparameters as the spot-check/calibration runs; if the Optuna
search (src/tune_head_a_optuna.py, run externally — see D3 report
Limitations) finds a better configuration, this script should be re-run
with the tuned params substituted before final submission.
"""

import json
import subprocess
import sys
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = Path("data/processed/gold_customer_features.csv")
TARGET = "retained_90d"
ID = "Customer ID"
CHOSEN_THRESHOLD = 0.80  # from docs/profit_threshold_sweep.csv, Extra Trees optimum

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET, ID])
y = df[TARGET]

for col in ["first_purchase_date", "last_purchase_date"]:
    X[col] = pd.to_datetime(X[col])
    X[col + "_days"] = (X[col] - pd.Timestamp("2009-12-01")).dt.days
    X = X.drop(columns=[col])

categorical = ["country"]
numeric = [c for c in X.columns if c not in categorical]
feature_order = list(X.columns)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

# Train on train+calibration slices (everything before the held-out test
# window), then wrap+calibrate on the calibration slice, matching
# calibrate_and_threshold.py's protocol — the held-out test slice remains
# untouched by this final fit, exactly as the D3 test-set-protection
# requirement expects.
order_col = "last_purchase_date_days"
X["_order"] = X[order_col]
ordered = X.sort_values("_order").index
n = len(ordered)
train_idx = ordered[: int(n * 0.70)]
calib_idx = ordered[int(n * 0.70): int(n * 0.85)]
test_idx = ordered[int(n * 0.85):]

X_train = X.loc[train_idx].drop(columns=["_order"])
X_calib = X.loc[calib_idx].drop(columns=["_order"])
X_test = X.loc[test_idx].drop(columns=["_order"])
y_train, y_calib, y_test = y.loc[train_idx], y.loc[calib_idx], y.loc[test_idx]

model = ExtraTreesClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
pipe.fit(X_train, y_train)

final_artifact = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
final_artifact.fit(X_calib, y_calib)

# Confirm the shipped artifact reproduces the reported held-out metric
test_proba = final_artifact.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, test_proba)
ap = average_precision_score(y_test, test_proba)
print(f"Artifact reproduces held-out test: ROC-AUC={roc:.4f}, PR-AUC={ap:.4f}")
print("(should match docs/final_heldout_test_results.csv for Extra Trees)")

Path("models").mkdir(exist_ok=True)
artifact_path = "models/head_a_retention_model.joblib"
joblib.dump(final_artifact, artifact_path)

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_commit = "unavailable (no git history in this checkout)"

metadata = {
    "model_version": "v0.3-d3",
    "training_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    "row_counts": {"train": len(X_train), "calibration": len(X_calib), "held_out_test": len(X_test)},
    "feature_names_in_order": feature_order,
    "categorical_features": categorical,
    "numeric_features": numeric,
    "library_versions": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
    "git_commit": git_commit,
    "decision_threshold": CHOSEN_THRESHOLD,
    "decision_rule": "target (send retention offer) if predict_proba(retain)[1] < threshold",
    "held_out_test_metric": {"roc_auc": roc, "pr_auc": ap, "metric_primary": "pr_auc"},
    "serialization": "joblib (pipeline + estimator + calibrator bundled as ONE artifact)",
    "known_risks": [
        "joblib uses pickle internally: loading this file executes arbitrary code if the "
        "file has been tampered with. Only load artifacts from the trusted model registry, "
        "never from an untrusted upload.",
        "The artifact is tied to the exact scikit-learn version above; a version mismatch "
        "between training and serving environments can silently change behavior or fail to "
        "load. The Dockerfile must pin scikit-learn to this version.",
        "Preprocessing (imputer/scaler/one-hot) is embedded inside this single artifact "
        "specifically so the API can never accidentally apply mismatched preprocessing logic "
        "written separately in serving code.",
    ],
}

with open("models/head_a_retention_model.metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\nSaved:\n  {artifact_path}\n  models/head_a_retention_model.metadata.json")
