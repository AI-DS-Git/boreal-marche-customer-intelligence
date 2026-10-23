"""
D3 — Head A spot-check: 9 model families vs. 2 required baselines, under
identical folds and identical preprocessing (spec 3.2 / D3 rubric).

Baselines:
  - DummyClassifier (prior)                       -> "no-skill" reference
  - Business-rule heuristic (recency <= 45 days)   -> "one-rule" reference,
    the kind of logic Boréal Marché could ship without any ML at all.

Model families (8, beyond the two baselines):
  Logistic Regression, Decision Tree, Random Forest, Extra Trees,
  AdaBoost, Gradient Boosting, Hist Gradient Boosting, K-Nearest Neighbors,
  Gaussian Naive Bayes.
  (XGBoost/LightGBM were considered but are unavailable in this environment
  — PyPI is not reachable from the build sandbox. HistGradientBoosting is
  scikit-learn's native equivalent and is included instead; see D3 report
  Limitations for the substitution rationale.)

Metrics reported (per spec): PR-AUC (primary), ROC-AUC, Brier score,
recall@top-decile, fit time.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA = Path("data/processed/gold_customer_features.csv")

df = pd.read_csv(DATA)

TARGET = "retained_90d"
ID = "Customer ID"

# Remove identifier and target
X = df.drop(columns=[TARGET, ID])
y = df[TARGET]

# Dates are converted to useful numeric features
for col in ["first_purchase_date", "last_purchase_date"]:
    X[col] = pd.to_datetime(X[col])
    X[col + "_days"] = (
        X[col] - pd.Timestamp("2009-12-01")
    ).dt.days
    X = X.drop(columns=[col])

categorical = ["country"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )
    ]
)


class RecencyRuleClassifier(BaseEstimator, ClassifierMixin):
    """One-rule business heuristic: predict 'will order again' if the
    customer's last purchase was within RECENCY_THRESHOLD days of the
    cutoff. This is the kind of rule Boréal Marché's ops team could apply
    today without any model, and is the second required baseline."""

    def __init__(self, recency_threshold=45, recency_col="recency_days"):
        self.recency_threshold = recency_threshold
        self.recency_col = recency_col

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def _score(self, X):
        # Softened step function so ties don't collapse rank-based metrics:
        # customers well inside the threshold score higher than those just
        # inside it, but the decision rule is still the single threshold.
        recency = X[self.recency_col].to_numpy(dtype=float)
        inside = recency <= self.recency_threshold
        margin = (self.recency_threshold - recency) / max(self.recency_threshold, 1)
        score = np.where(inside, 0.5 + 0.5 * np.clip(margin, 0, 1),
                          0.5 - 0.5 * np.clip(-margin, 0, 1))
        return score

    def predict_proba(self, X):
        p1 = self._score(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        recency = X[self.recency_col].to_numpy(dtype=float)
        return (recency <= self.recency_threshold).astype(int)


models = {
    "Dummy baseline": DummyClassifier(strategy="prior"),
    "Business-rule (recency<=45d)": RecencyRuleClassifier(recency_threshold=45),

    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=25),
    "Gaussian Naive Bayes": GaussianNB(),
}

# Chronological validation:
# earlier customers are training data;
# the latest observation cohort is validation data.
order_col = "last_purchase_date_days"
X["_order"] = X[order_col]
ordered = X.sort_values("_order").index

# Three expanding chronological folds
folds = [
    (0.60, 0.80),
    (0.70, 0.90),
    (0.80, 1.00)
]

results = []

for model_name, model in models.items():

    print()
    print("=" * 75)
    print(model_name)
    print("=" * 75)

    for fold_no, (train_end, valid_end) in enumerate(folds, 1):

        n = len(ordered)
        train_end_idx = int(n * train_end)
        valid_end_idx = int(n * valid_end)

        train_idx = ordered[:train_end_idx]
        valid_idx = ordered[train_end_idx:valid_end_idx]

        X_train = X.loc[train_idx].drop(columns=["_order"])
        X_valid = X.loc[valid_idx].drop(columns=["_order"])

        y_train = y.loc[train_idx]
        y_valid = y.loc[valid_idx]

        if model_name == "Business-rule (recency<=45d)":
            # No preprocessing/fitting needed — evaluate the raw rule
            # directly on the untransformed recency_days column so the
            # threshold stays in human-readable units.
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            probability = model.predict_proba(X_valid)[:, 1]
        else:
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", model)
            ])
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            probability = pipe.predict_proba(X_valid)[:, 1]

        roc = roc_auc_score(y_valid, probability)
        ap = average_precision_score(y_valid, probability)
        brier = brier_score_loss(y_valid, probability)

        # Recall at top decile: of the top 10% highest-scored customers,
        # what fraction of ALL actual positives in the fold do we catch?
        top_decile_n = max(1, int(np.ceil(0.10 * len(y_valid))))
        top_idx = np.argsort(-probability)[:top_decile_n]
        positives_total = y_valid.sum()
        recall_top_decile = (
            y_valid.to_numpy()[top_idx].sum() / positives_total
            if positives_total > 0 else np.nan
        )

        results.append({
            "model": model_name,
            "fold": fold_no,
            "roc_auc": roc,
            "average_precision": ap,
            "brier_score": brier,
            "recall_top_decile": recall_top_decile,
            "fit_time_sec": fit_time,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx)
        })

        print(
            f"Fold {fold_no}: "
            f"ROC-AUC={roc:.4f} | PR-AUC={ap:.4f} | Brier={brier:.4f} | "
            f"Recall@Decile={recall_top_decile:.4f} | FitTime={fit_time:.3f}s | "
            f"Train={len(train_idx):,} | Valid={len(valid_idx):,}"
        )

results_df = pd.DataFrame(results)

print()
print("=" * 75)
print("SPOT-CHECK SUMMARY (sorted by mean PR-AUC, primary metric)")
print("=" * 75)

summary = (
    results_df
    .groupby("model")
    .agg(
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
        mean_brier=("brier_score", "mean"),
        mean_recall_top_decile=("recall_top_decile", "mean"),
        mean_fit_time_sec=("fit_time_sec", "mean"),
    )
    .sort_values("mean_average_precision", ascending=False)
)

print(summary.to_string())

Path("docs").mkdir(exist_ok=True)

results_df.to_csv(
    "docs/head_a_spot_check_results.csv",
    index=False
)

summary.to_csv(
    "docs/head_a_spot_check_summary.csv"
)

print()
print("Saved:")
print("docs/head_a_spot_check_results.csv")
print("docs/head_a_spot_check_summary.csv")
