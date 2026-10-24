"""Weekly retraining + promotion gate for Head A (retention model).

Trains a CANDIDATE with the frozen D3 recipe (same chronological 70/15/15 split,
same tuned Extra Trees hyperparameters, isotonic calibration) on whatever
data/processed/gold_customer_features.csv currently contains, then evaluates
BOTH the candidate and the currently deployed artifact on the SAME held-out
slice and applies explicit promotion criteria.

It NEVER touches models/ unless --write-models is passed AND the gate passes
(the workflow does this on a branch and opens a pull request; a human merges,
and only the merge triggers CI -> Deploy). Nothing here deploys anything.

Outputs (always): reports/retrain/decision.json, reports/retrain/report.md
Outputs (if promoted + --write-models): models/head_a_retention_model.joblib,
  models/head_a_retention_model.metadata.json
Exit code is 0 either way; the decision is in decision.json ("promote": bool).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/gold_customer_features.csv")
MODELS = Path("models")
CURRENT_ART = MODELS / "head_a_retention_model.joblib"
CURRENT_META = MODELS / "head_a_retention_model.metadata.json"
OUT = Path("reports/retrain")
TARGET, ID = "retained_90d", "Customer ID"
THRESHOLD = 0.82  # frozen business threshold (docs/profit_threshold_sweep.csv); not re-tuned automatically
PARAMS = dict(n_estimators=450, max_depth=15, min_samples_leaf=27, min_samples_split=30,
              max_features=None, class_weight="balanced", random_state=42, n_jobs=-1)

# ---- Promotion criteria (all must hold) -----------------------------------
MIN_PR_AUC_GAIN = 0.005      # candidate must beat current PR-AUC by at least this (primary metric)
MAX_ROC_AUC_DROP = 0.01      # ...without losing more than this on ROC-AUC
MAX_BRIER_WORSE = 0.005      # ...or on calibration (Brier, lower is better)
MIN_ROWS_RATIO = 0.90        # ...and training data must not have shrunk >10% (data-loss guard)
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID])
    y = df[TARGET]
    for col in ["first_purchase_date", "last_purchase_date"]:
        X[col] = pd.to_datetime(X[col])
        X[col + "_days"] = (X[col] - pd.Timestamp("2009-12-01")).dt.days
        X = X.drop(columns=[col])
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    order = X.sort_values("last_purchase_date_days").index
    n = len(order)
    a, b = int(n * 0.70), int(n * 0.85)
    tr, ca, te = order[:a], order[a:b], order[b:]
    return (X.loc[tr], y.loc[tr]), (X.loc[ca], y.loc[ca]), (X.loc[te], y.loc[te])


def scores(model, X, y) -> dict:
    p = model.predict_proba(X)[:, 1]
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def build_candidate(Xtr, ytr, Xca, yca):
    categorical = ["country"]
    numeric = [c for c in Xtr.columns if c not in categorical]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipe = Pipeline([("preprocess", pre), ("model", ExtraTreesClassifier(**PARAMS))])
    pipe.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(Xca, yca)
    return cal, categorical, numeric


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-models", action="store_true", help="overwrite models/ if the gate passes")
    ap.add_argument("--force", action="store_true", help="treat the gate as passed (manual override, logged)")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    X, y = prepare(df)
    (Xtr, ytr), (Xca, yca), (Xte, yte) = split(X, y)
    cand, categorical, numeric = build_candidate(Xtr, ytr, Xca, yca)
    cand_s = scores(cand, Xte, yte)

    cur_meta = json.loads(CURRENT_META.read_text())
    cur = joblib.load(CURRENT_ART)
    cur_feats = cur_meta["feature_names_in_order"]
    schema_ok = list(X.columns) == cur_feats
    cur_s = scores(cur, Xte[cur_feats], yte) if schema_ok else None
    cur_rows = sum(cur_meta["row_counts"].values())
    rows_ratio = len(X) / cur_rows

    checks = {
        "schema_unchanged": schema_ok,
        "rows_not_shrunk": rows_ratio >= MIN_ROWS_RATIO,
    }
    if cur_s:
        checks["pr_auc_gain"] = cand_s["pr_auc"] - cur_s["pr_auc"] >= MIN_PR_AUC_GAIN
        checks["roc_auc_not_worse"] = cur_s["roc_auc"] - cand_s["roc_auc"] <= MAX_ROC_AUC_DROP
        checks["brier_not_worse"] = cand_s["brier"] - cur_s["brier"] <= MAX_BRIER_WORSE
    passed = all(checks.values())
    promote = passed or args.force

    version = "retrain-" + pd.Timestamp.now("UTC").strftime("%Y%m%d")
    decision = {
        "promote": promote, "gate_passed": passed, "forced": bool(args.force and not passed),
        "checks": checks, "candidate": cand_s, "current": cur_s,
        "rows": {"candidate": len(X), "current": cur_rows, "ratio": rows_ratio},
        "criteria": {"min_pr_auc_gain": MIN_PR_AUC_GAIN, "max_roc_auc_drop": MAX_ROC_AUC_DROP,
                     "max_brier_worse": MAX_BRIER_WORSE, "min_rows_ratio": MIN_ROWS_RATIO},
        "candidate_version": version, "current_version": cur_meta.get("model_version"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))

    def row(name, a, b, fmt="{:.4f}"):
        return f"| {name} | {fmt.format(a)} | {fmt.format(b) if b is not None else 'n/a'} |"

    lines = [
        f"## Weekly retraining report - `{version}`", "",
        f"**Decision: {'PROMOTE (open PR for human review)' if promote else 'KEEP CURRENT MODEL'}**"
        + (" - *forced override*" if decision["forced"] else ""), "",
        "Both models scored on the same held-out slice (latest 15% of customers by last purchase).", "",
        "| Metric | Candidate | Current |", "|---|---|---|",
        row("PR-AUC (primary)", cand_s["pr_auc"], cur_s and cur_s["pr_auc"]),
        row("ROC-AUC", cand_s["roc_auc"], cur_s and cur_s["roc_auc"]),
        row("Brier (lower better)", cand_s["brier"], cur_s and cur_s["brier"]),
        row("Customers", len(X), cur_rows, "{:d}"), "",
        "### Promotion gate", "", "| Check | Rule | Result |", "|---|---|---|",
        f"| schema_unchanged | feature list/order identical | {'PASS' if checks['schema_unchanged'] else 'FAIL'} |",
        f"| rows_not_shrunk | rows >= {MIN_ROWS_RATIO:.0%} of current | {'PASS' if checks['rows_not_shrunk'] else 'FAIL'} |",
    ]
    if cur_s:
        lines += [
            f"| pr_auc_gain | candidate - current >= {MIN_PR_AUC_GAIN} | {'PASS' if checks['pr_auc_gain'] else 'FAIL'} |",
            f"| roc_auc_not_worse | drop <= {MAX_ROC_AUC_DROP} | {'PASS' if checks['roc_auc_not_worse'] else 'FAIL'} |",
            f"| brier_not_worse | increase <= {MAX_BRIER_WORSE} | {'PASS' if checks['brier_not_worse'] else 'FAIL'} |",
        ]
    lines += ["", "**This PR does not deploy anything.** Merging it runs CI; the Deploy workflow then ships "
              "the new artifact. Closing the PR leaves production untouched.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))

    if promote and args.write_models:
        backup = MODELS / "previous"
        backup.mkdir(exist_ok=True)
        shutil.copy2(CURRENT_ART, backup / CURRENT_ART.name)
        shutil.copy2(CURRENT_META, backup / CURRENT_META.name)
        joblib.dump(cand, CURRENT_ART)
        meta = dict(cur_meta)
        meta.update({
            "model_version": version,
            "training_timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
            "row_counts": {"train": len(Xtr), "calibration": len(Xca), "held_out_test": len(Xte)},
            "feature_names_in_order": list(X.columns), "categorical_features": categorical,
            "numeric_features": numeric, "git_commit": git_sha(), "decision_threshold": THRESHOLD,
            "held_out_test_metric": {"roc_auc": cand_s["roc_auc"], "pr_auc": cand_s["pr_auc"],
                                     "brier": cand_s["brier"], "metric_primary": "pr_auc"},
            "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                                 "pandas": pd.__version__, "numpy": np.__version__,
                                 "joblib": joblib.__version__},
            "promotion": {"previous_version": cur_meta.get("model_version"), "previous_metrics": cur_s,
                          "forced": decision["forced"]},
        })
        CURRENT_META.write_text(json.dumps(meta, indent=2, default=str))
        print(f"Wrote candidate to {CURRENT_ART} (previous kept in {backup}/)")


if __name__ == "__main__":
    main()
