import pytest
from pydantic import ValidationError

from app.schemas import CustomerFeatures
from tests.helpers import golden, golden_features, model


def test_golden_prediction_regression():
    """A retrained or re-serialised model must not silently change this customer's score."""
    g = golden()
    p = float(model().predict_proba([golden_features()])[0])
    assert abs(p - g["expected_retain_probability"]) < 1e-5
    assert model().decide(p) == g["expected_decision"]


def test_decision_rule_boundary():
    m = model()
    assert m.threshold == 0.82
    assert m.decide(0.8199) == "target_with_offer"
    assert m.decide(0.82) == "no_action"


def test_missing_optional_values_are_imputed():
    f = golden()["features"].copy()
    f["price_std"] = None
    f["avg_days_between_orders"] = None
    p = float(model().predict_proba([CustomerFeatures(**f)])[0])
    assert 0.0 <= p <= 1.0


def test_batch_matches_single():
    f = golden_features()
    batch = model().predict_proba([f, f, f])
    assert len(batch) == 3 and abs(batch[0] - batch[2]) < 1e-12


def test_schema_rejects_unknown_and_inconsistent_fields():
    f = golden()["features"]
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**f, "surprise": 1})
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**f, "first_purchase_date": "2011-12-01", "last_purchase_date": "2010-01-01"})
    with pytest.raises(ValidationError):
        CustomerFeatures(**{**f, "weekend_transaction_ratio": 1.5})
