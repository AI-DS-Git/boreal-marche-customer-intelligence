"""Head A model wrapper: load once, validate, predict.

The artifact is ONE joblib file containing preprocessing + calibrated estimator, so
this code only has to put the 28 features in the recorded order. Nothing here
re-implements preprocessing.
"""
import json
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    def __init__(self, model_path: Path, metadata_path: Path, strict_versions: bool = False):
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
