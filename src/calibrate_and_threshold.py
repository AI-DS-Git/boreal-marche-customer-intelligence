"""
D3 — Calibration, profit-optimal threshold, final held-out test, and error
analysis for Head A (retention prediction).

Protocol (chronological, three non-overlapping slices by last_purchase_date):
  Train        0% - 70%   -> fit candidate models
  Calibration  70% - 85%  -> fit a probability calibrator on top of each model
  Held-out test 85% - 100% -> touched exactly once, for final reported metrics

Candidates carried forward from the spot-check (docs/head_a_spot_check_summary.csv):
  Extra Trees, Random Forest (ranked #1/#2 by mean PR-AUC).

NOTE ON TUNING: Optuna is unavailable in this build environment (PyPI is not
reachable). The models below use the same hyperparameters as the spot-check
(sensible defaults, not a leaderboard-chased configuration). The Optuna
search space is written separately in src/tune_head_a_optuna.py for the
student to run locally/in CI and splice the tuned hyperparameters back in
before final submission — see D3 report Limitations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss
)

DATA = Path("data/processed/gold_customer_features.csv")
TARGET = "retained_90d"
ID = "Customer ID"

# ------------------------------------------------------------------
# Cost assumptions (stated explicitly per D3 requirement — these are
# planning assumptions for a college capstone prototype, not verified
# Boréal Marché financials; D5 should revisit them if real figures surface).
# ------------------------------------------------------------------
COUPON_COST = 5.00          # CAD cost of a targeted retention offer
MARGIN_RATE = 0.20          # assumed gross margin on an order
RECOVERY_RATE = 0.30        # assumed probability a coupon "saves" a true churner

print("=" * 90)
print("COST ASSUMPTIONS (stated per D3 requirement)")
print("=" * 90)
print(f"Coupon cost per targeted customer:      ${COUPON_COST:.2f}")
print(f"Assumed gross margin rate on an order:   {MARGIN_RATE:.0%}")
print(f"Assumed coupon recovery rate:            {RECOVERY_RATE:.0%}")
print("(A targeted TRUE churner is assumed recovered with probability")
print(" RECOVERY_RATE, contributing MARGIN_RATE x their average order value.")
print(" A targeted customer who would have retained anyway costs the coupon")
print(" with no incremental benefit — pure waste.)")

# ------------------------------------------------------------------
# Load + prepare
# ------------------------------------------------------------------
df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET, ID])
y = df[TARGET]

for col in ["first_purchase_date", "last_purchase_date"]:
    X[col] = pd.to_datetime(X[col])
    X[col + "_days"] = (X[col] - pd.Timestamp("2009-12-01")).dt.days
    X = X.drop(columns=[col])

categorical = ["country"]
numeric = [c for c in X.columns if c not in categorical]

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

order_col = "last_purchase_date_days"
X["_order"] = X[order_col]
ordered = X.sort_values("_order").index
n = len(ordered)

train_idx = ordered[: int(n * 0.70)]
calib_idx = ordered[int(n * 0.70): int(n * 0.85)]
test_idx = ordered[int(n * 0.85):]

print(f"\nSplit sizes -> train: {len(train_idx):,} | calibration: {len(calib_idx):,} | "
      f"held-out test: {len(test_idx):,}")

X_train = X.loc[train_idx].drop(columns=["_order"])
X_calib = X.loc[calib_idx].drop(columns=["_order"])
X_test = X.loc[test_idx].drop(columns=["_order"])
y_train = y.loc[train_idx]
y_calib = y.loc[calib_idx]
y_test = y.loc[test_idx]

# revenue_per_order for the calibration/test customers, used in the profit model
revenue_per_order_test = df.loc[test_idx, "revenue_per_order"].fillna(
    df.loc[train_idx, "revenue_per_order"].median()
)

candidates = {
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

reliability_rows = []
final_test_rows = []
profit_rows = []
error_analysis_rows = []

for name, base_model in candidates.items():
    print(f"\n{'=' * 75}\n{name}\n{'=' * 75}")

    pipe = Pipeline([("preprocess", preprocessor), ("model", base_model)])
    pipe.fit(X_train, y_train)

    # Raw ("before calibration") probabilities on the held-out test set
    raw_test_proba = pipe.predict_proba(X_test)[:, 1]

    # Fit calibrator on the calibration slice, on top of the already-fitted
    # base pipeline. scikit-learn >= 1.6 replaced cv="prefit" with wrapping
    # the fitted estimator in FrozenEstimator; fall back to cv="prefit" on
    # older versions for portability across environments (e.g. SageMaker
    # training images may pin an older scikit-learn).
    if _HAS_FROZEN:
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    else:
        calibrated = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv="prefit")
    calibrated.fit(X_calib, y_calib)
    cal_test_proba = calibrated.predict_proba(X_test)[:, 1]

    # --- Reliability curves (before / after), evaluated on held-out test ---
    for label, proba in [("before_calibration", raw_test_proba), ("after_calibration", cal_test_proba)]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
        for mp, fp in zip(mean_pred, frac_pos):
            reliability_rows.append({
                "model": name, "stage": label,
                "mean_predicted": mp, "fraction_positive": fp
            })

    brier_before = brier_score_loss(y_test, raw_test_proba)
    brier_after = brier_score_loss(y_test, cal_test_proba)
    print(f"Brier score on held-out test — before calibration: {brier_before:.4f} | "
          f"after calibration: {brier_after:.4f}")

    # --- Final held-out test results (reported once) ---
    roc = roc_auc_score(y_test, cal_test_proba)
    ap = average_precision_score(y_test, cal_test_proba)
    top_decile_n = max(1, int(np.ceil(0.10 * len(y_test))))
    top_idx = np.argsort(-cal_test_proba)[:top_decile_n]
    recall_top_decile = y_test.to_numpy()[top_idx].sum() / y_test.sum()

    final_test_rows.append({
        "model": name, "roc_auc": roc, "pr_auc": ap,
        "brier_before_calibration": brier_before,
        "brier_after_calibration": brier_after,
        "recall_top_decile": recall_top_decile,
        "n_test": len(y_test),
    })
    print(f"HELD-OUT TEST (calibrated): ROC-AUC={roc:.4f} | PR-AUC={ap:.4f} | "
          f"Recall@Decile={recall_top_decile:.4f}")

    # --- Profit-optimal threshold sweep ---
    # Business rule: target (send coupon to) a customer if predicted P(retain) < t
    # i.e. the model flags them as at risk of churning.
    thresholds = np.linspace(0.05, 0.95, 37)
    y_test_arr = y_test.to_numpy()
    avg_margin_test = (revenue_per_order_test.to_numpy() * MARGIN_RATE)

    best_profit, best_t = -np.inf, None
    for t in thresholds:
        targeted = cal_test_proba < t  # flagged "at risk" -> send coupon
        n_targeted = targeted.sum()
        true_churners_targeted = targeted & (y_test_arr == 0)
        would_have_retained_targeted = targeted & (y_test_arr == 1)

        benefit = (avg_margin_test[true_churners_targeted] * RECOVERY_RATE).sum()
        cost = n_targeted * COUPON_COST
        profit = benefit - cost

        profit_rows.append({"model": name, "threshold": t, "n_targeted": int(n_targeted),
                             "expected_profit": profit})
        if profit > best_profit:
            best_profit, best_t = profit, t

    # Compare against blanket-coupon policy (current Boréal Marché practice:
    # everyone gets the coupon every quarter) and against "coupon nobody".
    blanket_profit = (avg_margin_test[y_test_arr == 0] * RECOVERY_RATE).sum() - len(y_test_arr) * COUPON_COST
    no_action_profit = 0.0

    print(f"Profit-optimal threshold: {best_t:.2f} -> expected profit ${best_profit:,.2f} "
          f"on {len(y_test_arr)} held-out customers")
    print(f"  vs. blanket coupon to everyone: ${blanket_profit:,.2f}")
    print(f"  vs. no action at all:           ${no_action_profit:,.2f}")

    # --- Error analysis by segment (using the chosen threshold) ---
    targeted_final = cal_test_proba < best_t
    predicted_label = (~targeted_final).astype(int)  # predict retain=1 if NOT targeted

    seg_df = df.loc[test_idx, ["Customer ID", "country", "monetary_total"]].copy()
    seg_df["actual"] = y_test_arr
    seg_df["predicted"] = predicted_label
    seg_df["correct"] = (seg_df["actual"] == seg_df["predicted"]).astype(int)
    seg_df["is_uk"] = (seg_df["country"] == "United Kingdom")
    seg_df["spend_tercile"] = pd.qcut(seg_df["monetary_total"], 3, labels=["low", "mid", "high"], duplicates="drop")

    for seg_col in ["is_uk", "spend_tercile"]:
        for seg_val, grp in seg_df.groupby(seg_col, observed=True):
            actual_pos = grp["actual"] == 1
            recall = (grp.loc[actual_pos, "predicted"] == 1).mean() if actual_pos.sum() else np.nan
            error_analysis_rows.append({
                "model": name, "segment_type": seg_col, "segment_value": str(seg_val),
                "n": len(grp), "accuracy": grp["correct"].mean(),
                "recall_positive_class": recall,
            })

# ------------------------------------------------------------------
# Save everything
# ------------------------------------------------------------------
Path("docs").mkdir(exist_ok=True)

pd.DataFrame(reliability_rows).to_csv("docs/calibration_reliability.csv", index=False)
pd.DataFrame(final_test_rows).to_csv("docs/final_heldout_test_results.csv", index=False)
pd.DataFrame(profit_rows).to_csv("docs/profit_threshold_sweep.csv", index=False)
pd.DataFrame(error_analysis_rows).to_csv("docs/error_analysis_by_segment.csv", index=False)

with open("docs/cost_assumptions.json", "w") as f:
    json.dump({
        "coupon_cost_cad": COUPON_COST,
        "margin_rate": MARGIN_RATE,
        "recovery_rate": RECOVERY_RATE,
    }, f, indent=2)

print("\nSaved:")
print("  docs/calibration_reliability.csv")
print("  docs/final_heldout_test_results.csv")
print("  docs/profit_threshold_sweep.csv")
print("  docs/error_analysis_by_segment.csv")
print("  docs/cost_assumptions.json")
