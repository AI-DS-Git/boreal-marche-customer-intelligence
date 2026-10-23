"""Runtime configuration. Every setting can be overridden by an environment variable.

Secrets (API keys, database passwords) are never committed: they come from the
environment or a local .env file. See .env.example for the full list.
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Artifacts
    model_path: Path = ROOT / "models" / "head_a_retention_model.joblib"
    model_metadata_path: Path = ROOT / "models" / "head_a_retention_model.metadata.json"
    rules_path: Path = ROOT / "docs" / "head_b_rules.json"
    popularity_path: Path = ROOT / "docs" / "head_b_popularity.json"
    products_path: Path = ROOT / "app" / "data" / "products.json"
    policies_path: Path = ROOT / "app" / "data" / "policies.json"
    drift_reference_path: Path = ROOT / "models" / "drift_reference.json"

    # Database for the prediction log. SQLite by default; Compose uses PostgreSQL.
    database_url: str = "sqlite:///" + str(ROOT / "data" / "api_logs.db")

    # Deployed version, injected at build time (see Dockerfile / CD workflow).
    git_sha: str = "dev"

    # Refuse to serve if scikit-learn differs from the version recorded in the model
    # metadata. Off by default so local development is forgiving; on in production.
    strict_library_versions: bool = False

    # Monitoring thresholds (traffic light): amber at warn, red at alert.
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    error_rate_alert: float = 0.05
    p95_latency_ms_alert: float = 500.0
    min_requests_for_drift: int = 30

    # Optional alert sink (e.g. a Slack/Discord/Teams incoming webhook URL).
    alert_webhook_url: Optional[str] = None

    # Assistant: similarity floor below which a policy question is refused.
    retrieval_floor: float = 0.4


def get_settings() -> Settings:
    return Settings()
