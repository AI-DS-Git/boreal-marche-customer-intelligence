"""
D3 — Optuna hyperparameter tuning for the top 2 Head A candidates
(Extra Trees, Random Forest), with all trials logged to MLflow.

*** THIS SCRIPT HAS NOT BEEN RUN. ***
It was written and reviewed in a build sandbox where PyPI is unreachable
(org network policy blocks it), so `optuna` and `mlflow` cannot be
installed there. Run it in your own environment where you have run D1/D2:

    pip install optuna mlflow
    mlflow ui --backend-store-uri ./mlruns          # to inspect afterwards, in another shell
    python src/tune_head_a_optuna.py

Design:
  - Bayesian search (Optuna's default TPE sampler) over each model's most
    impactful hyperparameters.
  - Pruning via the SAME 3 chronological folds used in spot_check_head_a.py:
    each fold's PR-AUC is reported to Optuna as an intermediate value
    (trial.report), and MedianPruner stops a trial early if fold 1 (or 2)
    is already worse than the median trial at that point. This works for
    any model, not just gradient-boosted ones with native early stopping.
  - Every trial (params + per-fold PR-AUC + mean PR-AUC + fit time) is
    logged to MLflow as a nested run under one parent run per model, so the
    full search history survives for the D3 "all trials logged" requirement.
  - Writes the winning params to docs/head_a_tuned_params.json so
    build_model_artifact.py can be re-run with them substituted in before
    final submission.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    import mlflow
except ImportError as e:
    raise SystemExit(
        "optuna and/or mlflow are not installed. This script is meant to run "
        "in your own environment (pip install optuna mlflow), not in the build "
        "sandbox used for the rest of D3 — see the module docstring."
    ) from e

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

DATA = Path("data/processed/gold_customer_features.csv")
TARGET = "retained_90d"
ID = "Customer ID"
N_TRIALS = 60

mlflow.set_experiment("boreal-marche-head-a-tuning")

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
folds = [(0.60, 0.80), (0.70, 0.90), (0.80, 1.00)]


def build_fold(train_end, valid_end):
    n = len(ordered)
    train_idx = ordered[: int(n * train_end)]
    valid_idx = ordered[int(n * train_end): int(n * valid_end)]
    return (X.loc[train_idx].drop(columns=["_order"]), y.loc[train_idx],
            X.loc[valid_idx].drop(columns=["_order"]), y.loc[valid_idx])


FOLD_DATA = [build_fold(*f) for f in folds]

SEARCH_SPACES = {
    "Extra Trees": lambda trial: ExtraTreesClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "Random Forest": lambda trial: RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
}


def make_objective(model_name, parent_run_id):
    def objective(trial):
        model = SEARCH_SPACES[model_name](trial)
        fold_scores = []
        t0 = time.perf_counter()

        with mlflow.start_run(run_name=f"{model_name}-trial-{trial.number}",
                               nested=True, parent_run_id=parent_run_id):
            mlflow.log_params(trial.params)
            mlflow.log_param("model_family", model_name)

            for fold_no, (X_tr, y_tr, X_va, y_va) in enumerate(FOLD_DATA, 1):
                pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_va)[:, 1]
                score = average_precision_score(y_va, proba)
                fold_scores.append(score)
                mlflow.log_metric("pr_auc", score, step=fold_no)

                trial.report(score, step=fold_no)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    raise optuna.TrialPruned()

            fit_time = time.perf_counter() - t0
            mean_score = float(np.mean(fold_scores))
            mlflow.log_metric("mean_pr_auc", mean_score)
            mlflow.log_metric("fit_time_sec", fit_time)

        return mean_score

    return objective


best_params_all = {}

for model_name in SEARCH_SPACES:
    print(f"\n{'=' * 75}\nTuning {model_name}\n{'=' * 75}")

    with mlflow.start_run(run_name=f"{model_name}-optuna-search") as parent_run:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        study.optimize(make_objective(model_name, parent_run.info.run_id), n_trials=N_TRIALS)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_mean_pr_auc", study.best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_pruned", sum(1 for t in study.trials
                                           if t.state == optuna.trial.TrialState.PRUNED))

    print(f"Best mean PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Trials: {len(study.trials)} total, "
          f"{sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)} pruned")

    best_params_all[model_name] = {"best_params": study.best_params, "best_mean_pr_auc": study.best_value}

Path("docs").mkdir(exist_ok=True)
with open("docs/head_a_tuned_params.json", "w") as f:
    json.dump(best_params_all, f, indent=2)

print("\nSaved: docs/head_a_tuned_params.json")
print("Next: splice these params into build_model_artifact.py's ExtraTreesClassifier(...) call")
print("(or re-run calibrate_and_threshold.py / build_model_artifact.py with them) before final submission.")
