## Model card — Head A (retention)
**Purpose.** Estimate the probability that a customer places another order within 90 days, so that staff can decide whether to review a retention offer. It supports a human decision; it does not act on its own.

**Model.** Tuned Extra Trees classifier with isotonic probability calibration, shipped as one artifact (preprocessing + model + calibrator). Version `v0.3-d3`, scikit-learn 1.8.0.

**Training data.** 5,249 customers (train 3,674 · calibration 787 · held-out test 788), 28 behavioural features built from transactions (recency, frequency, spend, product variety, timing, country flag).

**Performance (held-out test).** PR-AUC 0.875 · ROC-AUC 0.730. Decision rule: offer if P(retain) < 0.82, chosen to maximise expected profit (about $1,066 vs $201 for a blanket coupon on the test set, under stated cost assumptions).

**Limits.** Moderate ranking power (ROC-AUC 0.73). Trained on one retailer's 2009–2011 history, so behaviour may have shifted. A subgroup audit found that low-spend, mid-spend and new customers are never scored above about 0.52, so all of them fall below the 0.82 cut-off; this is why every offer is reviewed by a person. Not for automated, customer-facing decisions.

## Privacy notice
We use your purchase history to estimate whether you may stop buying and to suggest products. This is done by an automated tool. A staff member reviews every retention offer before it is sent, and the tool never makes a decision about you on its own. You may ask us which information was used and the main reasons for a result, to correct that information, and to have a person review any decision that concerns you. To exercise your rights, contact our privacy officer.

This prototype is built on a public, de-identified retail dataset. It does not collect personal information from visitors. Requests you send are logged (feature values, prediction, timing, request id) to monitor model quality; no names, emails or addresses are stored. The assistant refuses to share or look up another person's data. Under Quebec's Law 25, customers must be told when an automated tool informs a decision about them and may ask for a human review; a production deployment would add consent capture, retention limits and a named privacy officer.

## Data provenance
UCI Machine Learning Repository, *Online Retail II* (UK online retailer, December 2009 to December 2011, public license). Transactions were cleaned (cancellations and non-product lines removed), then aggregated into one row per customer. Recommendation rules were mined from basket data (351 rules after de-duplication and a lift filter). Policy texts in the assistant are prototype placeholders, not real company policy.

## Architecture
```
Streamlit UI ──HTTP──> FastAPI service ──> Head A model (joblib)
                         │   ├──────────> Head B rule engine (351 rules)
                         │   └──────────> Head C assistant (rule provider; LLM providers plug in)
                         └──> PostgreSQL: predictions, requests, alerts, delayed outcomes
Monitoring: PSI/KS drift, latency, errors, alerts (+ webhook), 90-day ground-truth backfill
CI/CD: GitHub Actions -> Docker image (tagged with commit SHA) -> AWS
```
