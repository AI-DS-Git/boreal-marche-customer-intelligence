"""Runtime configuration. Every setting can be overridden by an environment variable.

Secrets (API keys, database passwords) are never committed: they come from the
environment or a local .env file. See .env.example for the full list.
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


EMBEDDING_FLOORS = {"hashing": 0.25, "openai": 0.30, "ollama": 0.40}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

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
    log_retention_days: int = 120  # 90-day label delay + 30 days; see GOVERNANCE_LOI25.md (retention and deletion)

    # Optional alert sink (e.g. a Slack/Discord/Teams incoming webhook URL).
    alert_webhook_url: Optional[str] = None

    # Assistant retrieval. embedding_provider: lexical | hashing (offline, default) | openai | ollama.
    # retrieval_floor applies to the lexical index; embedding_floor (if set) overrides the
    # per-embedder default in EMBEDDING_FLOORS. The openai/ollama defaults are UNCALIBRATED:
    # run scripts/eval_assistant.py --calibrate against the real back-end and set EMBEDDING_FLOOR.
    embedding_provider: str = "hashing"
    retrieval_floor: float = 0.4
    embedding_floor: Optional[float] = None

    # LLM back-ends (assistant providers "openai" and "ollama"). A provider is enabled only
    # when its key / base URL is set; otherwise /chat answers "not configured".
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    ollama_base_url: Optional[str] = None
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "bge-m3"

    def floor_for(self, embedding_provider: str) -> float:
        if embedding_provider == "lexical":
            return self.retrieval_floor
        return self.embedding_floor if self.embedding_floor is not None else EMBEDDING_FLOORS[embedding_provider]


def get_settings() -> Settings:
    return Settings()
