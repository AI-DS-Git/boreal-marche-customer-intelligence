# Model card – Head A retention model (v0.3-d3)

*Boréal Marché Customer Intelligence Platform · Manish Chauhan · last updated 2026-10-26*

## Summary
Estimates the probability that a customer places at least one more order within 90 days. A calibrated Extra Trees
classifier served by the FastAPI service (`POST /predict`). The output supports a **staff decision**; it does not act on its own.

## Model details
| Item | Value |
|---|---|
| Version / trained | `v0.3-d3`, 2026-10-02 (UTC), code commit `9805a8b` |
| Algorithm | Extra Trees, 450 trees, max depth 15, min leaf 27, class-weight balanced; isotonic calibration on a separate calibration slice |
| Inputs | 28 customer-level purchase-behaviour features (spend, order frequency, recency, tenure, basket variety, seasonality, country) |
| Output | `retain_probability` in [0,1]; decision rule: **offer review if probability < 0.82** (threshold chosen on expected profit, `docs/profit_threshold_sweep.csv`) |
| Serving | One joblib artifact (preprocessing + model + calibrator); loaded once at start-up; scikit-learn 1.8.0 pinned |
| Explanations | `POST /explain`: top factor groups by occlusion on the uncalibrated score (staff-facing) |

## Intended use
- Rank customers for **human review** of a retention offer (coupon) in an online retail setting.
- Support monitoring and what-if exploration by staff.

## Out-of-scope uses
- Any decision made **without** human review (see `GOVERNANCE_LOI25.md`, design consequence).
- Pricing, credit, eligibility, employment or any decision with legal or similarly significant effects.
- Telling a customer *why* they were flagged: the explanation describes how the model reacts, not causes.
- Customers or markets unlike the training data (single UK retailer, 2009–2011; 91% UK).
- Customers with a very short history (1–2 orders) or low spend: scores are capped near 0.52 (see Fairness).

## Training and evaluation data
UCI Online Retail II, customer-level gold table (5,249 customers). Chronological 70/15/15 split by last purchase
date: 3,674 train, 787 calibration, 788 held-out (`data/processed`, see `DATA_CARD.md`).

## Performance (held-out, 788 customers)
| Metric | Value |
|---|---|
| PR-AUC (primary) | 0.8753 (metadata); 0.8726 recomputed by `scripts/fairness_audit.py` on the same split |
| ROC-AUC | 0.7301 (metadata); 0.7260 recomputed |
| Recall of retained / precision at 0.82 | 0.548 / 0.906 |
| Expected profit (D3 assumptions: $5 coupon, 20% margin, 30% recovery) | $1,065.65 vs $201.36 for a blanket coupon |

The held-out slice is the most recent period and is richer and older than the training slice (retention 76% vs 31%).
Real-world accuracy cannot be measured until labels mature 90 days after each prediction (first signal about 2027-01-24).

## Fairness and subgroup results (`reports/fairness/`, spec 3.7)
No protected attributes exist in the data; groups are region, tenure band and spend quartile. Recall of retained customers
by group at 0.82: spend quartiles Q1–Q3 = **0.0** (the model never scores them above 0.52), Q4 = 0.71; tenure 0–90 days = 0.0,
91–365 days = 0.21, 366+ days = 0.78; UK 0.56, Europe (non-UK) 0.46 (n=78, wide interval). Six groups are flagged
(>5 points below overall recall of 0.55). Mitigations were tested and are reported in the D5 report, including two that did **not** help.

## Limitations and known risks
- Calibration is isotonic (a step function): many inputs share one probability. Rank within a group, do not over-read small differences.
- Subgroup sizes are small (46–125 customers in the weakest groups); intervals are wide.
- Population shift between training and held-out periods; the weekly retraining job and drift monitor watch for it.
- `joblib` artifacts execute code on load: only load files from the trusted repository (see metadata `known_risks`).
- `country`/`is_uk` add nothing measurable (PR-AUC 0.8732 without vs 0.8726 with) and are candidates for removal.

## Maintenance
Weekly retraining workflow with a promotion gate (PR-AUC gain ≥ 0.005, ROC-AUC drop ≤ 0.01, Brier worsening ≤ 0.005);
human merge required; CI then deploys. Drift monitor: PSI amber 0.10 / red 0.25.
