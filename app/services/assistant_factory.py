"""Builds the assistant from settings: retrieval index plus the configured providers.

Lives outside main.py so tests and scripts/eval_assistant.py can build the same assistant
without importing FastAPI.
"""
import json
import logging
from typing import Any, Dict, Optional

from app.services.assistant import AssistantService, Provider, RuleBaselineProvider
from app.services.llm import LLMClient, LLMError, build_client
from app.services.llm_provider import LLMProvider, build_policy_index
from app.services.retrieval import HashingEmbedder

log = logging.getLogger("boreal.api")


def _log(**kw: Any) -> None:
    log.info(json.dumps(kw, default=str))


def build_assistant(s, rules) -> AssistantService:
    """Retrieval index (lexical, offline hashing, or a real embedding model) plus the providers
    that are configured. An unreachable embedding back-end falls back to the offline index at
    start-up (logged) rather than failing the container."""
    clients = {k: build_client(k, openai_key=s.openai_api_key, openai_base=s.openai_base_url,
                               openai_chat=s.openai_chat_model, openai_embed=s.openai_embed_model,
                               ollama_base=s.ollama_base_url, ollama_chat=s.ollama_chat_model,
                               ollama_embed=s.ollama_embed_model) for k in ("openai", "ollama")}
    kind = s.embedding_provider
    embedder: Optional[Any] = None
    if kind == "hashing":
        embedder = HashingEmbedder()
    elif kind in ("openai", "ollama"):
        embedder = clients.get(kind)
        if embedder is None:
            _log(event="embedding_fallback", wanted=kind, reason="not configured", using="hashing")
            kind, embedder = "hashing", HashingEmbedder()
    try:
        policies = build_policy_index(s.policies_path, embedder)
    except LLMError as exc:
        _log(event="embedding_fallback", wanted=kind, reason=str(exc), using="hashing")
        kind, embedder = "hashing", HashingEmbedder()
        policies = build_policy_index(s.policies_path, embedder)
    floor = s.floor_for(kind)
    rule = RuleBaselineProvider(policies, rules.recommend, rules.describe, floor)
    providers: Dict[str, Provider] = {"rule": rule}
    for name, client in clients.items():
        if isinstance(client, LLMClient):
            providers[name] = LLMProvider(client, policies, rules.recommend, rules.describe, rule, floor)
    _log(event="assistant_ready", retrieval=kind, floor=floor, providers=sorted(providers))
    return AssistantService(providers)
