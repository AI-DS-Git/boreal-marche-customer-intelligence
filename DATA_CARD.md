# Data card – customer retention dataset

*Boréal Marché Customer Intelligence Platform · Manish Chauhan · last updated 2026-10-26*

## Source and licence
- **UCI Online Retail II**: transactions of a UK-based online retailer, December 2009 – December 2011 (public dataset).
  Licence as stated on the UCI dataset page (CC BY 4.0 at time of writing; verify before redistribution). Raw files are **not** committed (`.gitignore`); `data/README.md` explains how to obtain and checksum them.
- No data was collected from real Boréal Marché customers.

## Composition
| Stage | Rows | Notes |
|---|---|---|
| Bronze (raw, both sheets) | 1,067,371 | as loaded |
| Silver (cleaned) | 1,007,913 | exact duplicates, cancellations, non-positive quantities, zero/invalid prices removed (`docs/cleaning_pipeline_results.csv`) |
| Gold (customer table) | 5,249 customers | one row per customer; 28 model features; label `retained_90d` |

- Label: 1 if the customer places an order within 90 days after the observation date; overall 42.7% retained.
  Training slice 31%, calibration 65%, held-out 76% (chronological split, so later customers are more established).
- Countries: 41; United Kingdom 91.2% (4,787), Germany 91, France 75, Spain 34, Belgium 26.
- Spend and tenure are strongly skewed; 47% of customers have 1–2 orders.

## Personal information
- No names, e-mails, addresses or payment data. `Customer ID` is a pseudonymous key from the public dataset.
- `country` is a coarse location; it is **not needed** by the model (ablation in the D5 report) and is a candidate for removal.
- The API logs request feature values, the prediction and timing (no identity fields) for monitoring; retained 120 days (see `GOVERNANCE_LOI25.md`).

## Collection and processing
Point-in-time features are computed from transactions before the observation date; the label from the 90 days after, to avoid
leakage (`docs/leakage_*.csv`). Features are aggregated per customer; no free text is used by the model.

## Known biases and limits
- One retailer, one country (UK-dominated), two years; may not transfer to Québec customers.
- No demographic attributes: fairness can only be checked on behaviour and country, not on protected grounds.
- Low-spend and short-history customers are rarely classed as retained by the model (see model card).
- Seasonal effects (holiday periods) are strong; the label window may straddle the holiday season.

## Maintenance
Retraining reads the same table through `scripts/retrain.py`; a data-quality test (`tests/test_data_quality.py`) and the monitor's
null/out-of-range checks guard the inputs.
