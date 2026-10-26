"""Head A model wrapper: load once, validate, predict.

The artifact is ONE joblib file containing preprocessing + calibrated estimator, so
this code only has to put the 28 features in the recorded order. Nothing here
re-implements preprocessing.
"""
import json
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.schemas import CustomerFeatures

EPOCH = date(2009, 12, 1)  # the training code measured purchase dates from this day


class ModelVersionMismatch(RuntimeError):
    pass


def features_to_row(f: CustomerFeatures) -> Dict[str, Any]:
    row = f.model_dump()
    first = row.pop("first_purchase_date")
    last = row.pop("last_purchase_date")
    row["first_purchase_date_days"] = (first - EPOCH).days
    row["last_purchase_date_days"] = (last - EPOCH).days
    # Null stays NaN so the pipeline's median imputer handles it.
    for k in ("price_std", "avg_days_between_orders"):
        if row[k] is None:
            row[k] = np.nan
    return row


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path, strict_versions: bool = False,
                 reference_path: Optional[Path] = None):
        with open(metadata_path, encoding="utf-8") as fh:
            self.metadata: Dict[str, Any] = json.load(fh)
        self.feature_order: List[str] = self.metadata["feature_names_in_order"]
        self.threshold: float = float(self.metadata["decision_threshold"])
        self.model_version: str = self.metadata["model_version"]

        trained_with = self.metadata.get("library_versions", {}).get("scikit-learn")
        self.version_warning = None
        if trained_with and trained_with != sklearn.__version__:
            self.version_warning = (
                f"model trained with scikit-learn {trained_with}, serving with {sklearn.__version__}"
            )
            if strict_versions:
                raise ModelVersionMismatch(self.version_warning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(model_path)
        self._set_single_threaded()
        self._baseline = self._load_baseline(reference_path)

    def frame(self, items: Sequence[CustomerFeatures]) -> pd.DataFrame:
        df = pd.DataFrame([features_to_row(i) for i in items])
        return df[self.feature_order]

    def predict_proba(self, items: Sequence[CustomerFeatures]) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict_proba(self.frame(items))[:, 1]

    def decide(self, p_retain: float) -> str:
        # Decision rule from training: target a customer with a retention offer when the
        # calibrated probability of being retained is BELOW the profit-optimal threshold.
        return "target_with_offer" if p_retain < self.threshold else "no_action"


    # ---- per-prediction explanation ----------------------------------------------------
    def _set_single_threaded(self) -> None:
        """The forest was saved with n_jobs=-1, which starts a thread pool on EVERY predict call. Measured on a
        2-vCPU machine: single-row predict 112 ms -> 32 ms, 1,000-row batch 173 ms -> 62 ms, with identical outputs
        (scripts/perf_profile.py). Concurrency comes from the web server's workers instead."""
        try:
            self._raw_model().named_steps["model"].n_jobs = 1
        except Exception:
            pass  # unknown pipeline layout: keep the saved setting

    @staticmethod
    def _load_baseline(path: Optional[Path]) -> Dict[str, Any]:
        """A 'typical customer': the median of each numeric feature in the training reference sample
        (models/drift_reference.json) and the most common country. Empty if the file is missing."""
        if path is None or not Path(path).exists():
            return {}
        feats = json.loads(Path(path).read_text(encoding="utf-8")).get("features", {})
        base: Dict[str, Any] = {k: float(np.median(v["sample"])) for k, v in feats.items() if v.get("sample")}
        base["country"] = "United Kingdom"
        base["is_uk"] = 1
        return base

    # Correlated predictors are occluded TOGETHER, otherwise credit is split arbitrarily between near-duplicates
    # (e.g. spend, quantity and revenue per order) and many single-feature effects tie exactly.
    GROUPS = {
        "recency": ["recency_days", "last_purchase_date_days"],
        "order frequency": ["frequency_orders", "transaction_count", "active_days", "active_months",
                            "orders_per_active_month", "avg_days_between_orders"],
        "spend and volume": ["monetary_total", "monetary_mean_order", "revenue_per_order", "total_quantity",
                             "average_quantity", "quantity_per_order"],
        "basket variety": ["unique_products", "unique_product_descriptions", "products_per_order", "avg_lines_per_order"],
        "price level": ["average_unit_price", "price_std"],
        "tenure": ["first_purchase_date_days", "customer_lifetime_days"],
        "seasonality and weekends": ["holiday_transactions", "holiday_season_ratio", "weekend_transactions",
                                     "weekend_transaction_ratio"],
        "country": ["country", "is_uk"],
    }

    def _raw_model(self):
        """The uncalibrated pipeline inside the calibrated model. Isotonic calibration is a step function, so
        many different inputs map to the same calibrated value and group effects tie; the raw score is graded.
        Falls back to the calibrated model if the sklearn internals differ."""
        try:
            return self.model.calibrated_classifiers_[0].estimator.estimator
        except Exception:
            return self.model

    def explain(self, item: CustomerFeatures, k: int = 3) -> List[Dict[str, Any]]:
        """Group-occlusion explanation: set each group of related predictors to the typical-customer value, re-score,
        and report how much the retention probability moves. Effects are changes in the UNCALIBRATED model score (graded);
        positive = this customer's values RAISE retention relative to a typical customer. It describes how the model reacts, not causes in the
        customer's life."""
        if not self._baseline:
            return []
        row = self.frame([item])
        names = [g for g, members in self.GROUPS.items() if any(m in row.columns for m in members)]
        df = pd.concat([row] * (len(names) + 1), ignore_index=True)
        df = df.astype({c: float for c in df.columns if c != "country"})  # medians are fractional
        for i, name in enumerate(names, start=1):
            for m in self.GROUPS[name]:
                if m in self._baseline and m in df.columns:
                    df.loc[i, m] = self._baseline[m]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = self._raw_model().predict_proba(df[self.feature_order])[:, 1]
        out: List[Dict[str, Any]] = []
        for i, name in enumerate(names, start=1):
            effect = float(p[0] - p[i])
            if abs(effect) < 1e-4:
                continue
            detail = ", ".join(
                f"{m}={row.iloc[0][m] if isinstance(row.iloc[0][m], str) else round(float(row.iloc[0][m]), 2)}"
                for m in self.GROUPS[name][:3] if m in row.columns)
            out.append({"group": name, "detail": detail, "effect": round(effect, 4),
                        "direction": "raises_retention" if effect > 0 else "lowers_retention"})
        out.sort(key=lambda d: abs(d["effect"]), reverse=True)
        return out[:k]
