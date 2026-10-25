"""OpenAI and Ollama providers end to end over real sockets against a local fake server:
tool-calling loop, citations, answer verifier, guardrails, unreachable back-end, true streaming."""
import json
import time

import pytest

from app.config import get_settings
from app.services.assistant import RuleBaselineProvider
from app.services.llm import LLMError, OllamaClient, OpenAIClient
from app.services.llm_provider import LLMProvider, verify_answer
from app.services.retrieval import EmbeddingIndex
from tests.fake_llm import FakeLLM
from tests.helpers import rules

pytestmark = pytest.mark.parametrize("kind", ["openai", "ollama"])


def client_for(kind, srv):
    if kind == "openai":
        return OpenAIClient("test-key", base_url=srv.openai_base, timeout=5)
    return OllamaClient(srv.ollama_base, timeout=5)


def provider(kind, srv, floor=0.25):
    c = client_for(kind, srv)
    idx = EmbeddingIndex(get_settings().policies_path, c)  # embeddings also go through the client
    rs = rules()
    rule = RuleBaselineProvider(idx, rs.recommend, rs.describe, floor)
    return LLMProvider(c, idx, rs.recommend, rs.describe, rule, floor)


def tool_results(messages):
    return [m for m in messages if m["role"] == "tool"]


def grounded_script(query):
    """Round 1: call policy_search. Round 2: answer with the text the tool returned."""
    def script(messages, tools):
        res = tool_results(messages)
        if not res:
            return [("tool", "policy_search", {"query": query})]
        rows = json.loads(res[-1]["content"])
        return [("token", rows[0]["text"][:60]), ("token", rows[0]["text"][60:])]
    return script


def test_tool_loop_answers_from_tool_result_with_citation(kind):
    with FakeLLM(grounded_script("return refund")) as srv:
        r = provider(kind, srv).answer("What is your return policy?", "en")
    assert not r.refused and r.provider == kind
    assert r.citations and r.citations[0]["source_id"] == "returns"
    assert [t["tool"] for t in r.tool_calls] == ["policy_search"]
    assert "30 days" in r.answer


def test_invented_product_code_is_replaced_by_verifier_fallback(kind):
    script = lambda m, t: [("token", "Try product 99999Z, it pairs well.")]  # noqa: E731
    with FakeLLM(script) as srv:
        r = provider(kind, srv).answer("Recommend something that goes with 85123A", "en")
    assert "99999Z" not in r.answer
    assert r.tool_calls[-1]["tool"] == "verifier_fallback"


def test_invented_probability_is_replaced_by_verifier_fallback(kind):
    def script(messages, tools):
        if not tool_results(messages):
            return [("tool", "retention_info", {})]
        return [("token", "Your churn probability is 73%.")]
    with FakeLLM(script) as srv:
        r = provider(kind, srv).answer("What is the churn probability for my account?", "en")
    assert "73%" not in r.answer
    assert r.tool_calls[-1]["tool"] == "verifier_fallback"


def test_model_that_skips_tools_is_not_trusted(kind):
    script = lambda m, t: [("token", "Returns are accepted within 90 days.")]  # noqa: E731
    with FakeLLM(script) as srv:
        r = provider(kind, srv).answer("What is your return policy?", "en")
    assert "90 days" not in r.answer
    assert r.tool_calls[-1]["tool"] == "verifier_fallback"


def test_recommendation_codes_come_from_the_tool(kind):
    def script(messages, tools):
        res = tool_results(messages)
        if not res:
            return [("tool", "recommendations", {"basket": ["85123A"], "k": 2})]
        codes = [row["code"] for row in json.loads(res[-1]["content"])]
        return [("token", "You may also like " + " and ".join(codes) + ".")]
    with FakeLLM(script) as srv:
        r = provider(kind, srv).answer("Recommend something that goes with 85123A", "en")
    assert r.tool_calls[0]["tool"] == "recommendations" and not r.refused
    assert not any(t["tool"] == "verifier_fallback" for t in r.tool_calls)


def test_no_evidence_is_refused_not_guessed(kind):
    def script(messages, tools):
        if not tool_results(messages):
            return [("tool", "policy_search", {"query": "qqqq zzzz"})]
        return [("token", "Yes, we have a student discount of 15%.")]
    with FakeLLM(script) as srv:
        r = provider(kind, srv, floor=0.5).answer("Do you offer a student discount?", "en")
    assert r.refused and r.refusal_reason == "no_evidence"


def test_privacy_and_out_of_scope_never_reach_the_model(kind):
    with FakeLLM(lambda m, t: [("token", "should not be called")]) as srv:
        p = provider(kind, srv)
        a = p.answer("What is the email of customer 12345?", "en")
        b = p.answer("Tell me a joke about cats", "en")
        assert srv.chat_calls == 0
    assert a.refusal_reason == "privacy" and b.refusal_reason == "out_of_scope"


def test_unreachable_backend_is_reported_not_raised(kind):
    with FakeLLM(lambda m, t: []) as srv:
        p = provider(kind, srv)
    # server is now shut down
    r = p.answer("What is your return policy?", "en")
    assert r.refused and r.refusal_reason == "provider_unavailable"


def test_tokens_stream_before_the_answer_is_complete(kind):
    def script(messages, tools):
        if not tool_results(messages):
            return [("tool", "policy_search", {"query": "return"})]
        return [("token", "Items can be returned "), ("sleep", 0.4), ("token", "within 30 days.")]
    with FakeLLM(script) as srv:
        p = provider(kind, srv)
        t0, first = time.perf_counter(), None
        for ev in p.stream("What is your return policy?", "en"):
            if ev["type"] == "token" and first is None:
                first = time.perf_counter() - t0
            if ev["type"] == "final":
                total = time.perf_counter() - t0
    assert first is not None and first < total - 0.3, (first, total)


def test_clients_raise_llmerror_on_bad_url(kind):
    c = OpenAIClient("k", base_url="file:///etc/passwd") if kind == "openai" else OllamaClient("file:///etc/passwd")
    with pytest.raises(LLMError):
        c.embed(["x"])


def test_verify_answer_unit(kind):
    assert verify_answer("See 85123A.", {"85123A"}, "") == []
    assert verify_answer("See 85123A.", set(), "") != []
    assert verify_answer("Fee is 5%.", set(), "a 5% fee") == []
    assert verify_answer("Fee is 7%.", set(), "a 5% fee") != []
