"""LLM-backed assistant provider (OpenAI or Ollama) with typed tools and hard guardrails.

The model never answers from memory. It may only (1) call the four typed tools below and
(2) phrase their results. Around the model sit deterministic checks:

  before : privacy and out-of-scope guardrails (pre_guard) - the model never sees these messages
  during : at most MAX_ROUNDS tool rounds; the final round has no tools, forcing a text answer
  after  : verify_answer() - every product code in the answer must come from a tool result (or
           the user's own message), and every percentage or decimal must appear verbatim in a
           tool result. If the check fails, or the model skipped the tools, the answer is replaced
           with the deterministic rule-provider answer and a `verifier_fallback` trace is added.

Spec 3.4 hard rule: product codes only from the Head B tool, probabilities only from the Head A API.
"""
import json
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from app.services.assistant import (MESSAGES, PRODUCT_CODE, ChatResult, PolicyIndex, Provider, pre_guard)
from app.services.llm import LLMClient, LLMError

MAX_ROUNDS = 3
NUMBER_CLAIM = re.compile(r"\d+(?:[.,]\d+)?\s*%|\b0[.,]\d+\b")

TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "policy_search",
        "description": "Search Boreal Marche policies (returns, shipping, coupons, privacy, data, "
                       "price adjustment, contact). Use for any policy or how-do-I question.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "recommendations",
        "description": "Product suggestions for a basket of product codes. The ONLY source of product codes.",
        "parameters": {"type": "object", "properties": {
            "basket": {"type": "array", "items": {"type": "string"}, "description": "product codes, e.g. 85123A"},
            "k": {"type": "integer", "description": "how many suggestions (1-5)"}}, "required": ["basket"]}}},
    {"type": "function", "function": {
        "name": "product_lookup",
        "description": "Describe one product from its code.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "retention_info",
        "description": "Use when the user asks about churn, retention or the probability they will stop buying. "
                       "The assistant cannot compute this in chat.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = {
    "en": "You are the Boreal Marche customer assistant. Answer in English, in at most three sentences. "
          "Use ONLY tool results; never answer a policy or product question from memory. Use policy_search "
          "for policies, recommendations for product suggestions, product_lookup for one product code, "
          "retention_info for churn or retention questions. Never state a probability, percentage or "
          "product code that is not in a tool result. If policy_search returns NO_EVIDENCE, say you "
          "cannot find it and that a staff member will follow up.",
    "fr": "Vous etes l'assistant client de Boreal Marche. Repondez en francais, en trois phrases au plus. "
          "Utilisez UNIQUEMENT les resultats des outils; ne repondez jamais de memoire a une question sur "
          "les politiques ou les produits. Utilisez policy_search pour les politiques, recommendations "
          "pour les suggestions, product_lookup pour un code produit, retention_info pour les questions "
          "de depart ou de retention. N'indiquez jamais de probabilite, de pourcentage ou de code produit "
          "absent d'un resultat d'outil. Si policy_search renvoie NO_EVIDENCE, dites que vous ne trouvez "
          "pas l'information et qu'un membre du personnel fera le suivi.",
}


def verify_answer(answer: str, allowed_codes: Set[str], allowed_text: str) -> List[str]:
    """Returns the reasons the answer must be rejected (empty list = it passes)."""
    problems = []
    for code in PRODUCT_CODE.findall(answer):
        if code.upper() not in allowed_codes:
            problems.append(f"product code {code} not from a tool")
    for claim in NUMBER_CLAIM.findall(answer):
        if claim.replace(" ", "") not in allowed_text.replace(" ", ""):
            problems.append(f"number {claim} not from a tool")
    return problems


class ToolBox:
    """Runs the typed tools for ONE conversation turn and remembers what they returned."""

    def __init__(self, policies: Any, recommend: Callable[[List[str], int], List[dict]],
                 describe: Callable[[str], Optional[str]], floor: float, lang: str, user_message: str):
        self.policies, self.recommend, self.describe = policies, recommend, describe
        self.floor, self.lang = floor, lang
        self.allowed_codes: Set[str] = {c.upper() for c in PRODUCT_CODE.findall(user_message)}
        self.allowed_text = user_message
        self.citations: List[dict] = []
        self.traces: List[dict] = []
        self.evidence = False
        self.no_evidence = False

    def run(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "policy_search":
                out, summary = self._policy(str(args.get("query", "")))
            elif name == "recommendations":
                out, summary = self._reco(args)
            elif name == "product_lookup":
                out, summary = self._lookup(str(args.get("code", "")))
            elif name == "retention_info":
                out, summary = MESSAGES["retention_tool"][self.lang], "redirected to the Prediction page"
                self.evidence = True
            else:
                out, summary = f"ERROR unknown tool {name}", "unknown tool"
        except (TypeError, ValueError, AttributeError) as e:
            out, summary = f"ERROR bad arguments: {type(e).__name__}", "bad arguments"
        self.allowed_text += "\n" + out
        self.traces.append({"tool": name, "arguments": args, "result_summary": summary})
        return out

    def _policy(self, query: str) -> Tuple[str, str]:
        hits = [h for h in self.policies.search(query, 2) if h["score"] >= self.floor]
        if not hits:
            self.no_evidence = True
            return "NO_EVIDENCE", "no policy above the similarity floor"
        self.evidence = True
        top = hits[0]
        title = top["policy"]["title_" + self.lang]
        self.citations.append({"source_id": top["policy"]["id"], "title": title, "score": round(top["score"], 3)})
        rows = [{"id": h["policy"]["id"], "title": h["policy"]["title_" + self.lang],
                 "text": h["policy"]["text_" + self.lang]} for h in hits]
        return json.dumps(rows, ensure_ascii=False), f"top hit {top['policy']['id']} ({top['score']:.2f})"

    def _reco(self, args: Dict[str, Any]) -> Tuple[str, str]:
        basket = [str(c).upper() for c in args.get("basket", []) if PRODUCT_CODE.fullmatch(str(c))]
        if not basket:
            return "ERROR basket needs product codes like 85123A", "no valid codes"
        k = max(1, min(5, int(args.get("k", 5))))
        recs = self.recommend(basket, k)
        self.evidence = True
        self.allowed_codes |= {str(r["item"]).upper() for r in recs} | set(basket)
        rows = [{"code": r["item"], "description": r.get("description"), "because_of": r.get("because_of")}
                for r in recs]
        return json.dumps(rows, ensure_ascii=False), f"{len(recs)} items"

    def _lookup(self, code: str) -> Tuple[str, str]:
        code = code.strip().upper()
        desc = self.describe(code) if PRODUCT_CODE.fullmatch(code) else None
        self.evidence = True
        if desc:
            self.allowed_codes.add(code)
            return json.dumps({"code": code, "description": desc}), desc
        return "NOT_FOUND", "not found"


class LLMProvider(Provider):
    def __init__(self, client: LLMClient, policies: Any, recommend: Callable[[List[str], int], List[dict]],
                 describe: Callable[[str], Optional[str]], fallback: Provider, floor: float):
        self.client, self.name = client, client.name
        self.policies, self.recommend, self.describe = policies, recommend, describe
        self.fallback, self.floor = fallback, floor

    def answer(self, message: str, language: str) -> ChatResult:
        result = None
        for ev in self.stream(message, language):
            if ev["type"] == "final":
                result = ev["result"]
        assert result is not None
        return result

    def _final(self, text: str, lang: str, box: ToolBox, refused: Optional[str] = None) -> ChatResult:
        return ChatResult(answer=text, language=lang, provider=self.name,
                          citations=box.citations if not refused else [], tool_calls=box.traces,
                          refused=bool(refused), refusal_reason=refused)

    def stream(self, message: str, language: str) -> Iterator[Dict[str, Any]]:
        guard = pre_guard(message, language, self.name)
        if guard:
            yield {"type": "token", "text": guard.answer}
            yield {"type": "final", "result": guard}
            return
        box = ToolBox(self.policies, self.recommend, self.describe, self.floor, language, message)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM[language]},
                                          {"role": "user", "content": message}]
        text = ""
        try:
            for rnd in range(MAX_ROUNDS):
                text = ""
                calls: List[Dict[str, Any]] = []
                for ev in self.client.chat_stream(messages, TOOLS if rnd < MAX_ROUNDS - 1 else []):
                    if ev["type"] == "token":
                        text += ev["text"]
                        yield {"type": "token", "text": ev["text"]}
                    elif ev["type"] == "tool_call":
                        calls.append(ev)
                if not calls:
                    break
                if text:
                    yield {"type": "replace", "text": ""}  # text said alongside a tool call is not the answer
                    text = ""
                messages.append(self.client.assistant_tool_message(calls))
                for c in calls:
                    content = box.run(c["name"], c["arguments"])
                    yield {"type": "tool", **box.traces[-1]}
                    messages.append(self.client.tool_message(c, content))
        except LLMError:
            r = ChatResult(answer=MESSAGES["provider_unavailable"][language], language=language,
                           provider=self.name, refused=True, refusal_reason="provider_unavailable")
            yield {"type": "replace", "text": r.answer}
            yield {"type": "final", "result": r}
            return

        text = text.strip()
        if box.no_evidence and not box.evidence:
            r = self._final(MESSAGES["no_evidence"][language], language, box, refused="no_evidence")
            yield {"type": "replace", "text": r.answer}
            yield {"type": "final", "result": r}
            return
        problems = ["model gave no answer"] if not text else []
        if text and not box.traces:
            problems.append("model skipped the tools")
        if text:
            problems += verify_answer(text, box.allowed_codes, box.allowed_text)
        if problems:
            fb = self.fallback.answer(message, language)
            fb.provider = self.name
            fb.tool_calls = box.traces + [{"tool": "verifier_fallback", "arguments": {},
                                           "result_summary": "; ".join(problems)[:200]}]
            yield {"type": "replace", "text": fb.answer}
            yield {"type": "final", "result": fb}
            return
        yield {"type": "final", "result": self._final(text, language, box)}


def build_policy_index(path, embedder: Optional[Any]) -> Any:
    """Embedding index when an embedder is configured, otherwise the lexical baseline."""
    if embedder is None:
        return PolicyIndex(path)
    from app.services.retrieval import EmbeddingIndex
    return EmbeddingIndex(path, embedder)
