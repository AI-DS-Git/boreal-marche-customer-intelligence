"""Data-quality suite: the shipped sample population and the drift reference must be valid
and mutually consistent. Runs in CI so a bad data or reference change fails the build."""
import json

from app.config import get_settings
from app.schemas import CustomerFeatures
from tests.helpers import model

SAMPLE = json_path = None


def _sample():
    from pathlib import Path
    p = Path(__file__).parent / "golden" / "sample_customers.json"
    return json.loads(p.read_text(encoding="utf-8"))["customers"]


def test_every_sample_customer_passes_the_api_schema():
    rows = _sample()
    assert len(rows) >= 500
    for r in rows:
        CustomerFeatures(**r)


def test_sample_has_variety_and_few_nulls():
    rows = _sample()
    assert len({r["monetary_total"] for r in rows}) > 300           # not a copied single customer
    null_share = sum(r["price_std"] is None for r in rows) / len(rows)
    assert null_share < 0.5
    assert all(r[k] is not None for r in rows for k in ("recency_days", "frequency_orders", "country"))


def test_drift_reference_covers_every_numeric_model_feature():
    ref = json.loads(get_settings().drift_reference_path.read_text(encoding="utf-8"))
    numeric = set(model().metadata["numeric_features"])
    assert numeric <= set(ref["features"]), numeric - set(ref["features"])
    assert ref["n"] > 1000 and "prediction" in ref


def test_model_feature_order_is_complete():
    m = model()
    assert len(m.feature_order) == 28 and m.threshold == 0.82
