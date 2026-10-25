"""Scores an assistant against the 40-question golden set (tests/golden/assistant_golden.json).

Three independent checks per question:
  tool     the first tool the assistant used equals expected_tool ("none" for refusals)
  refusal  refusal_reason equals expected_refusal (None = must answer)
  cite     for policy questions, the first citation is the expected policy
Used by the CI test (rule provider, offline) and by scripts/eval_assistant.py (any provider)."""
import json
from pathlib import Path
from typing import Any, Dict, List

GOLDEN_PATH = Path(__file__).parent / "golden" / "assistant_golden.json"


def load_golden() -> List[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def evaluate(service: Any, provider: str = "rule") -> Dict[str, Any]:
    rows, failures = [], []
    for g in load_golden():
        r = service.chat(g["question"], g["lang"], provider)
        tool = r.tool_calls[0]["tool"] if r.tool_calls else "none"
        row = {
            "id": g["id"], "lang": g["lang"], "tool": tool == g["expected_tool"],
            "refusal": r.refusal_reason == g["expected_refusal"],
            "cite": (not g["expected_policy"]) or (bool(r.citations) and r.citations[0]["source_id"] == g["expected_policy"]),
        }
        rows.append(row)
        if not (row["tool"] and row["refusal"] and row["cite"]):
            failures.append({"id": g["id"], "question": g["question"], "got_tool": tool,
                             "got_refusal": r.refusal_reason, "expected_tool": g["expected_tool"],
                             "expected_refusal": g["expected_refusal"], "expected_policy": g["expected_policy"],
                             "citations": [c["source_id"] for c in r.citations], "answer": r.answer[:120]})
    n = len(rows)
    pol = [g for g in load_golden() if g["expected_policy"]]
    return {
        "n": n,
        "tool_choice": sum(r["tool"] for r in rows) / n,
        "refusal_correct": sum(r["refusal"] for r in rows) / n,
        "citation_correct": sum(r["cite"] for r in rows if r["id"] in {g["id"] for g in pol}) / len(pol),
        "by_lang": {lang: sum(r["tool"] and r["refusal"] and r["cite"] for r in rows if r["lang"] == lang)
                    / sum(1 for r in rows if r["lang"] == lang) for lang in ("en", "fr")},
        "failures": failures,
    }
