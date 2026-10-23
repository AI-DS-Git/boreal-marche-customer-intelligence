"""
D3 — Script-mode training entry point for Head A: runs IDENTICALLY whether
invoked locally or as a SageMaker training job (spec 3.5 requirement).

Local:
    python src/train.py --n-estimators 300 --max-depth 8 --min-samples-leaf 5

SageMaker (via the SDK, see launch_sagemaker_hpo.py):
    SageMaker sets SM_CHANNEL_TRAIN (the S3-synced input channel directory)
    and SM_MODEL_DIR (where anything written is uploaded back to S3) as
    environment variables, and passes hyperparameters as --flag value CLI
    args automatically — this script's argparse defaults fall back to local
    paths so nothing changes for a local run.

*** NOT RUN IN THE BUILD SANDBOX *** — no AWS credentials for the student's
own account are available there, and this script's local-mode correctness
depends on data/processed/gold_customer_features.csv already existing
(produced earlier by build_gold_features.py, which WAS run and verified).
The training logic itself (ExtraTreesClassifier + same preprocessing as
calibrate_and_threshold.py) is the same code path already validated in this
sandbox — only the SageMaker plumbing (env vars, model upload) is untested
here because no such job was launched.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters (SageMaker HPO passes these as --flag value)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=2)

    # Script-mode contract: SM_CHANNEL_TRAIN / SM_MODEL_DIR are set by
    # SageMaker automatically; default to local paths so nothing else
    # about this script changes between environments.
    parser.add_argument("--train", type=str,
                         default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed"))
    parser.add_argument("--model-dir", type=str,
                         default=os.environ.get("SM_MODEL_DIR", "models"))
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.train) / "gold_customer_features.csv"
    df = pd.read_csv(data_path)

    TARGET = "retained_90d"
    ID = "Customer ID"
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
    train_idx = ordered[: int(n * 0.80)]
    valid_idx = ordered[int(n * 0.80):]

    X_train = X.loc[train_idx].drop(columns=["_order"])
    X_valid = X.loc[valid_idx].drop(columns=["_order"])
    y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_valid)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_valid, proba),
        "pr_auc": average_precision_score(y_valid, proba),
    }
    print(json.dumps(metrics))  # SageMaker HPO reads the objective metric from stdout via regex

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_dir / "model.joblib")
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
