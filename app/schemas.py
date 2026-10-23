"""Pydantic v2 request/response schemas. Extra fields are rejected everywhere."""
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CustomerFeatures(Strict):
    """The 28 Head A predictors, in business terms. Dates are ISO dates; the service
    converts them to 'days since 2009-12-01' exactly as training did."""

    recency_days: int = Field(ge=0, le=5000)
    frequency_orders: int = Field(ge=1)
    monetary_total: float = Field(ge=0)
    monetary_mean_order: float = Field(ge=0)
    average_quantity: float = Field(ge=0)
    total_quantity: int = Field(ge=0)
    unique_products: int = Field(ge=0)
    unique_product_descriptions: int = Field(ge=0)
    average_unit_price: float = Field(ge=0)
    country: str = Field(min_length=2, max_length=60)
    first_purchase_date: date
    last_purchase_date: date
    active_days: int = Field(ge=1)
    active_months: int = Field(ge=1)
    weekend_transactions: int = Field(ge=0)
    transaction_count: int = Field(ge=1)
    # Structurally undefined for single-line / single-order customers; the model's
    # preprocessing imputes the training median when these are null.
    price_std: Optional[float] = Field(default=None, ge=0)
    holiday_transactions: int = Field(ge=0)
    avg_days_between_orders: Optional[float] = Field(default=None, ge=0)
    customer_lifetime_days: int = Field(ge=0)
    orders_per_active_month: float = Field(ge=0)
    products_per_order: float = Field(ge=0)
    revenue_per_order: float = Field(ge=0)
    quantity_per_order: float = Field(ge=0)
    weekend_transaction_ratio: float = Field(ge=0, le=1)
    avg_lines_per_order: float = Field(ge=0)
    holiday_season_ratio: float = Field(ge=0, le=1)
    is_uk: Literal[0, 1]

    @model_validator(mode="after")
    def _dates_in_order(self):
        if self.last_purchase_date < self.first_purchase_date:
            raise ValueError("last_purchase_date must not be before first_purchase_date")
        return self


class PredictRequest(Strict):
    features: CustomerFeatures


class PredictResponse(Strict):
    request_id: str
    retain_probability: float
    churn_risk: float
    decision: Literal["target_with_offer", "no_action"]
    threshold: float
    model_version: str
    latency_ms: float


class BatchPredictRequest(Strict):
    items: List[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(Strict):
    request_id: str
    predictions: List[PredictResponse]
    count: int


class RecommendRequest(Strict):
    basket: List[str] = Field(default_factory=list, max_length=50)
    k: int = Field(default=5, ge=1, le=20)


class Recommendation(Strict):
    item: str
    description: Optional[str] = None
    confidence: Optional[float] = None
    lift: Optional[float] = None
    because_of: Optional[str] = None
    because_of_description: Optional[str] = None
    strategy: Literal["rule", "popularity"]


class RecommendResponse(Strict):
    request_id: str
    recommendations: List[Recommendation]
    latency_ms: float


class ChatRequest(Strict):
    message: str = Field(min_length=1, max_length=1000)
    language: Literal["auto", "en", "fr"] = "auto"
    provider: Literal["rule", "openai", "ollama"] = "rule"


class Citation(Strict):
    source_id: str
    title: str
    score: float


class ToolCall(Strict):
    tool: str
    arguments: dict
    result_summary: str


class ChatResponse(Strict):
    request_id: str
    answer: str
    language: Literal["en", "fr"]
    provider: str
    citations: List[Citation]
    tool_calls: List[ToolCall]
    refused: bool
    refusal_reason: Optional[str] = None
    latency_ms: float


class VersionResponse(Strict):
    git_sha: str
    model_version: str
    model_git_commit: Optional[str] = None
    decision_threshold: float
    rules_count: int
    sklearn_version: str


class HealthResponse(Strict):
    status: Literal["ok"]


class ReadyResponse(Strict):
    status: Literal["ready", "not_ready"]
    checks: dict


class ErrorResponse(Strict):
    request_id: Optional[str] = None
    error: str
    detail: Optional[str] = None
