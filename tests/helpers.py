"""Shared builders. Plain functions (not fixtures) so tests also run without pytest."""
import json
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.schemas import CustomerFeatures
from app.services.assistant import AssistantService, PolicyIndex, RuleBaselineProvider
from app.services.model_service import ModelService
from app.services.rules_service import RulesService

GOLDEN = Path(__file__).parent / "golden" / "customer_12347.json"


def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def golden_features() -> CustomerFeatures:
    return CustomerFeatures(**golden()["features"])


@lru_cache(maxsize=1)
def model() -> ModelService:
    s = get_settings()
    return ModelService(s.model_path, s.model_metadata_path)


@lru_cache(maxsize=1)
def rules() -> RulesService:
    s = get_settings()
    return RulesService(s.rules_path, s.popularity_path, s.products_path)


@lru_cache(maxsize=1)
def assistant() -> AssistantService:
    s = get_settings()
    prov = RuleBaselineProvider(PolicyIndex(s.policies_path), rules().recommend, rules().describe, s.retrieval_floor)
    return AssistantService({"rule": prov})
