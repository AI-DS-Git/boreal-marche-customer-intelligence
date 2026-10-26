# Governance – Québec Law 25

*Boréal Marché Customer Intelligence Platform · Manish Chauhan · last updated 2026-10-26*

**Scope.** This system is a prototype on a public, de-identified dataset; it holds no real customers' personal information. This file
states what a production deployment must do and which of those controls the prototype already implements.

## 0. Primary sources consulted (2026-10-26)
- *Act respecting the protection of personal information in the private sector* (CQLR c. P-39.1), Légis Québec: **s. 12.1**
  (decision based exclusively on automated processing), **s. 8.1** (technologies with profiling functions; "profilage" = collecting and
  using personal information to assess characteristics of a person, including preferences, interests or behaviour), **s. 23**
  (destroy or anonymise personal information when the purposes are achieved).
- Commission d'accès à l'information (CAI) page "Utilisation et communication de renseignements personnels", subsection on informing
  people about automated decisions (the CAI notes that its pages are plain-language summaries and the Act prevails).
- Obligations under s. 12.1, as the sources state them: inform the person **no later than when the decision is communicated**;
  **on request**, tell them which personal information was used, the **reasons and main factors and parameters** that led to the decision,
  and their right to have the information corrected; and give them the **opportunity to submit observations to a staff member able to review the decision**.
- This file is not legal advice; a real deployment needs review by the organisation's privacy officer.

## 1. Automated-decision register
| # | Decision point | Decided by | Personal information used | Principal factors | Human review | How a customer exercises rights |
|---|---|---|---|---|---|---|
| 1 | Retention-risk score (probability of no order in 90 days) | Model (`/predict`) | Purchase history aggregated to 28 features; country | Order frequency, spend and volume, tenure, seasonality (global importance, D5 report); per prediction via `/explain` | Staff see score and factor groups | Notice + request explanation, correction, human review |
| 2 | Retention offer (score < 0.82) | **Staff**, informed by #1 | Same | Same | **Mandatory: a staff member approves every offer before it is sent** | Observations to the reviewing staff member; contact the privacy officer |
| 3 | Product recommendations | Model rules (`/recommend`) | Items in the current basket only | Association-rule lift | Not required (low impact; ignorable) | Ignore; no profile stored |
| 4 | Assistant answers | Assistant (rule/LLM provider) | The question; policy corpus | Retrieved policy passages (cited) | Hand-off to staff when no evidence; privacy questions refused | Contact staff |

## 2. Design consequence
Section 12.1 obligations apply to a decision based **exclusively** on automated processing. To avoid issuing a fully automated
decision about a person, the design inserts a **human approval step** before any retention offer (register #2): the API returns
`target_with_offer` as a *recommendation*, the interface says "Send a retention offer (a staff member reviews it)", and the assistant's
policy text promises review. This is why the human step exists. A rubber-stamp review would not satisfy the intent of the law: the
reviewer must see the factors (`/explain`) and may override. We also treat the score as **profiling** (s. 8.1) and disclose it.
The fairness audit shows why review matters: 100% of low-spend and new customers fall below the 0.82 cut-off, so an unreviewed
rule would send them all an offer.

## 3. Customer notice (rendered in the application, About page)
**English.** We use your purchase history to estimate whether you may stop buying and to suggest products. This is done by an automated
tool. A staff member reviews every retention offer before it is sent, and the tool never makes a decision about you on its own. You may ask us
which information was used and the main reasons for a result, to correct that information, and to have a person review any decision that concerns you.
We collect only what is needed, limit staff access, and never sell personal information. To exercise your rights, contact our privacy officer.

**Français.** Nous utilisons votre historique d'achats pour estimer si vous risquez de cesser d'acheter et pour vous suggérer des produits.
Cette évaluation est faite par un outil automatisé. Un membre du personnel examine chaque offre de fidélisation avant son envoi, et l'outil ne prend
jamais seul une décision à votre sujet. Vous pouvez nous demander quels renseignements ont été utilisés et les principaux motifs d'un résultat, faire corriger
ces renseignements et faire réviser par une personne toute décision vous concernant. Nous ne recueillons que ce qui est nécessaire, limitons l'accès du personnel
et ne vendons jamais de renseignements personnels. Pour exercer vos droits, communiquez avec notre responsable de la protection des renseignements personnels.

## 4. Data-minimisation audit (28 model features)
| Feature group (features) | Necessary? | Evidence | Decision |
|---|---|---|---|
| Order frequency (`frequency_orders`, `transaction_count`, `active_days`, `active_months`, `orders_per_active_month`, `avg_days_between_orders`) | Yes | Largest permutation importance (PR-AUC −0.044 when shuffled) | Keep |
| Spend and volume (`monetary_total`, `monetary_mean_order`, `revenue_per_order`, `total_quantity`, `average_quantity`, `quantity_per_order`) | Yes | −0.008 | Keep; near-duplicates: consider dropping `revenue_per_order` and `monetary_mean_order` (identical values in the golden case) |
| Tenure (`first_purchase_date_days`, `customer_lifetime_days`) | Yes | −0.005 | Keep |
| Seasonality (`holiday_*`, `weekend_*`) | Marginal | −0.004 | Keep for now; review at retraining |
| Basket variety (`unique_products`, `unique_product_descriptions`, `products_per_order`, `avg_lines_per_order`) | Marginal | −0.001 | Candidate to remove |
| Price level (`average_unit_price`, `price_std`) | No measurable value | −0.001 (within noise) | Candidate to remove |
| Recency (`recency_days`, `last_purchase_date_days`) | Weak on the held-out slice | 0.000 (little recency spread in a chronological hold-out) | Keep; business-critical, re-test on a rolling window |
| **Country** (`country`, `is_uk`) | **No** | Without them: PR-AUC 0.8732 vs 0.8726, recall 0.548 with vs 0.559 without (`reports/fairness/ablation_country.json`); permutation importance 0.000 | **Remove at next retraining** (reduces proxy and re-identification risk) |
Logs store feature values but no names, e-mails or addresses.

## 5. Retention and deletion policy
| Data | Purpose | Retention | Deletion |
|---|---|---|---|
| Prediction log (feature values, score, decision) | Drift monitoring; join with labels | **120 days** (90-day label delay + 30 days grace) | Implemented: `LogStore.purge_older_than(120)` runs at every service start (setting `LOG_RETENTION_DAYS`); a scheduled job is the production path |
| Request log (endpoint, status, latency) | Operations metrics | 120 days | Same purge |
| Outcome labels (request id, 0/1) | Accuracy history | Kept (no features); delete with the customer's request on erasure | Manual until a request-id lookup is added |
| Drift reference (training distribution) | Monitoring | Life of the model version | Replaced at promotion |
| Raw/gold data | Training | While the purpose lasts; then destroy or anonymise (s. 23) | Not stored in the repository |
Current prototype state: the live API stores logs in a container-local SQLite file, which is also wiped at every redeploy. A production
deployment would use a managed database with encryption, access control and a named privacy officer.

## 6. Residual risks and open items
- No real consent capture or privacy-officer contact in the prototype.
- Explanations (`/explain`) describe how the model reacts, not causes; they are staff-facing. Giving them verbatim to a customer would overstate certainty (D5 report).
- Fairness gaps (spend, tenure) are documented with mitigations; none is deployed yet.
- Erasure by request id is not implemented.
