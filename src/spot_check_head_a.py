import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

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

models = {
    "Dummy baseline": DummyClassifier(
        strategy="prior"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# Chronological validation:
# earlier customers are training data;
# the latest observation cohort is validation data.
#
# We use the customer-level dataset ordered by their
# most recent purchase date before the cutoff.

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

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)

        probability = pipe.predict_proba(X_valid)[:, 1]

        roc = roc_auc_score(y_valid, probability)
        ap = average_precision_score(y_valid, probability)

        results.append({
            "model": model_name,
            "fold": fold_no,
            "roc_auc": roc,
            "average_precision": ap,
            "train_rows": len(train_idx),
            "validation_rows": len(valid_idx)
        })

        print(
            f"Fold {fold_no}: "
            f"ROC-AUC={roc:.4f} | "
            f"PR-AUC={ap:.4f} | "
            f"Train={len(train_idx):,} | "
            f"Valid={len(valid_idx):,}"
        )

results_df = pd.DataFrame(results)

print()
print("=" * 75)
print("SPOT-CHECK SUMMARY")
print("=" * 75)

summary = (
    results_df
    .groupby("model")
    .agg(
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std")
    )
    .sort_values("mean_roc_auc", ascending=False)
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
