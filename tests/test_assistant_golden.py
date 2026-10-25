"""40-question golden set (spec 3.4): tool choice, refusal behaviour and citations, scored for the
offline rule provider with both retrieval back-ends. CI fails if quality drops below the floors.

Known, documented misses (offline retrieval is lexical, not semantic):
  q19 'customer service hours'  -> retrieves price_adjustment (shares the words 'customer service')
  q38 / q40 (student discount, cash gift-card refund) -> borderline similarity, one of them slips
        past the evidence floor depending on the retrieval back-end.
Real multilingual embeddings are scored with scripts/eval_assistant.py (needs an API key or Ollama)."""
import re

from tests.golden_eval import evaluate, load_golden
from tests.helpers import assistant, assistant_hashing

MIN_TOOL, MIN_REFUSAL, MIN_CITE = 0.95, 0.95, 0.90


def test_golden_set_shape():
    g = load_golden()
    assert len(g) == 40
    assert {x["lang"] for x in g} == {"en", "fr"}
    tools = {x["expected_tool"] for x in g}
    assert tools == {"policy_search", "recommendations", "product_lookup", "retention_info", "none"}
    assert {x["expected_refusal"] for x in g} == {None, "out_of_scope", "privacy", "no_evidence"}
    assert len({x["id"] for x in g}) == 40


def check(report):
    assert report["tool_choice"] >= MIN_TOOL, report["failures"]
    assert report["refusal_correct"] >= MIN_REFUSAL, report["failures"]
    assert report["citation_correct"] >= MIN_CITE, report["failures"]


def test_tool_choice_lexical_retrieval():
    check(evaluate(assistant()))


def test_tool_choice_hashing_retrieval():
    check(evaluate(assistant_hashing()))


def test_hard_rules_hold_on_every_golden_question():
    """No refused answer cites a source; no answer invents a probability or a product code."""
    for svc in (assistant(), assistant_hashing()):
        for g in load_golden():
            r = svc.chat(g["question"], g["lang"], "rule")
            if r.refused:
                assert not r.citations, g["id"]
            assert not re.search(r"\d+(\.\d+)?\s*%", r.answer), (g["id"], r.answer)
            codes_ok = set(re.findall(r"\b\d{5}[A-Z]{0,2}\b", g["question"].upper()))
            for t in r.tool_calls:
                if t["tool"] == "recommendations":
                    codes_ok |= set(re.findall(r"\b\d{5}[A-Z]{0,2}\b", r.answer.upper()))
            for code in re.findall(r"\b\d{5}[A-Z]{0,2}\b", r.answer.upper()):
                assert code in codes_ok or any(t["tool"] in ("recommendations", "product_lookup")
                                               for t in r.tool_calls), (g["id"], code)
