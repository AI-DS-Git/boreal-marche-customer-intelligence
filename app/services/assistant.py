"""Head C skeleton: the rule/keyword baseline provider, guardrails and typed tools.

Spec 3.4 asks for three interchangeable providers behind one interface (rule baseline,
hosted LLM, local model). This file implements the interface and the RULE BASELINE only;
the OpenAI and Ollama providers plug into the same `Provider` interface later.
Retrieval here is lexical (token coverage), NOT the multilingual embedding index the
spec ultimately requires -- it exists so the end-to-end flow, the guardrails and the
citations can be built and tested now.

Hard rules (spec 3.4): the assistant never produces a probability or a product
identifier from its own text generation. Product codes and recommendations come only
from the Head B tool; retention probabilities only from the Head A API.
"""
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

PRODUCT_CODE = re.compile(r"\b(\d{5}[A-Za-z]{0,2})\b")

_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "do", "i", "my", "me", "can", "you", "how", "what",
    "and", "or", "for", "in", "on", "it", "we", "your", "does", "if", "will", "be", "about", "with",
    "le", "la", "les", "un", "une", "des", "de", "du", "est", "sont", "je", "mon", "ma", "mes",
    "puis", "peut", "comment", "que", "quoi", "et", "ou", "pour", "dans", "sur", "il", "nous",
    "votre", "vos", "si", "au", "aux", "en", "quel", "quelle", "quels", "quelles", "y",
    "combien", "coute", "cout", "tell", "please", "svp", "bonjour", "hello", "hi", "joke", "long", "much",
}
_FR_HINTS = {
    "le", "la", "les", "des", "une", "est", "sont", "je", "mon", "mes", "puis", "comment", "pour",
    "vous", "votre", "quel", "quelle", "retour", "livraison", "remboursement", "fidélité", "avec",
    "combien", "quand", "où", "pourquoi", "bonjour", "merci", "produit", "client", "données",
}

_DOMAIN = {
    "return", "refund", "ship", "shipping", "delivery", "coupon", "loyalty", "privacy", "data",
    "policy", "order", "price", "product", "recommend", "suggest", "retention", "churn", "customer",
    "staff", "contact", "service", "offer", "tracking", "delete", "deletion", "exchange", "discount",
    "retour", "remboursement", "livraison", "expedition", "fidelite", "confidentialite", "donnees",
    "politique", "commande", "prix", "produit", "recommande", "suggere", "client",
    "personnel", "offre", "suivi", "suppression", "echange", "rabais",
}

OTHER_CUSTOMER = re.compile(
    r"(customer|client|account|compte)\s*(id|number|numero|no\.?|#)?\s*:?\s*\d{4,}"
    r"|another customer|someone else|other customer|autre client|quelqu'un d'autre|"
    r"(email|courriel|address|adresse|phone|telephone)\s+(of|de|du|d')\s+",
    re.IGNORECASE,
)

RETENTION_WORDS = {"churn", "risque", "risk", "probabilite", "probability", "likelihood", "score"}
RECO_STEMS = ("recommend", "sugg", "goes", "pair", "match")  # sugg: suggest, suggere, suggerez...

MESSAGES = {
    "out_of_scope": {
        "en": "I can only help with Boréal Marché policies, products, recommendations and retention questions.",
        "fr": "Je peux seulement répondre aux questions sur les politiques, les produits, les recommandations et la rétention de Boréal Marché.",
    },
    "privacy": {
        "en": "I can't share or look up another customer's personal data. A staff member can help with identity-verified requests.",
        "fr": "Je ne peux pas communiquer ni consulter les données personnelles d'un autre client. Un membre du personnel peut traiter les demandes avec vérification d'identité.",
    },
    "no_evidence": {
        "en": "I couldn't find this in our policies, so I won't guess. I'm handing your question to a staff member (Monday to Friday, 9:00 to 17:00 ET).",
        "fr": "Je n'ai pas trouvé cette information dans nos politiques, alors je ne vais pas deviner. Je transmets votre question à un membre du personnel (du lundi au vendredi, de 9 h à 17 h HE).",
    },
    "retention_tool": {
        "en": "I can't compute a retention probability in chat. Use the Prediction page: it calls the retention model with the customer's features, and a staff member reviews any offer.",
        "fr": "Je ne peux pas calculer de probabilité de rétention dans le clavardage. Utilisez la page Prédiction : elle appelle le modèle avec les caractéristiques du client, et un membre du personnel révise toute offre.",
    },
    "provider_unavailable": {
        "en": "That assistant provider is not reachable right now; please try the rule provider.",
        "fr": "Ce fournisseur d'assistant est inaccessible pour le moment; veuillez essayer le fournisseur par règles.",
    },
    "not_implemented": {
        "en": "That provider is not configured on this server; use the rule provider.",
        "fr": "Ce fournisseur n'est pas configuré sur ce serveur; utilisez le fournisseur par règles.",
    },
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokens(s: str) -> List[str]:
    s = strip_accents(s.lower())
    out = []
    for t in re.findall(r"[a-z0-9]+", s):
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]  # crude plural stripping, enough for a lexical baseline
        out.append(t)
    return out


def _in_domain(message: str) -> bool:
    """True if any word in the message starts like a known domain word (so that
    'retourner' matches 'retour' and 'shipped' matches 'ship')."""
    words = [strip_accents(w) for w in re.findall(r"[a-zA-Zàâçéèêëîïôûùüÿœ]+", message.lower())]
    for w in words:
        for d in _DOMAIN:
            stem = d[:5] if len(d) > 5 else d
            if w.startswith(stem):
                return True
    return False


def detect_language(text: str) -> str:
    low = text.lower()
    if re.search(r"[àâçéèêëîïôûùüÿœ]", low):
        return "fr"
    words = set(re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", low))
    return "fr" if len(words & _FR_HINTS) >= 2 else "en"


def pre_guard(message: str, lang: str, provider: str) -> Optional["ChatResult"]:
    """Deterministic guardrails that run BEFORE any model sees the message (privacy first,
    then out-of-scope). Used by every LLM-backed provider."""
    norm = strip_accents(message.lower())
    if OTHER_CUSTOMER.search(norm):
        reason = "privacy"
    elif (not PRODUCT_CODE.search(message) and not _in_domain(message)
          and not (set(tokens(message)) & RETENTION_WORDS)):
        reason = "out_of_scope"
    else:
        return None
    return ChatResult(answer=MESSAGES[reason][lang], language=lang, provider=provider,
                      refused=True, refusal_reason=reason)


@dataclass
class ChatResult:
    answer: str
    language: str
    provider: str
    citations: List[dict] = field(default_factory=list)
    tool_calls: List[dict] = field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None


class Provider(ABC):
    name: str

    @abstractmethod
    def answer(self, message: str, language: str) -> ChatResult: ...

    def stream(self, message: str, language: str) -> Iterator[Dict[str, Any]]:
        """Events: {"type": "tool", ...trace}, {"type": "token", "text"}, {"type": "replace", "text"},
        then exactly one {"type": "final", "result": ChatResult}. The rule provider has the whole
        answer up front, so its tokens are word-chunked; LLM providers override this with the
        model's own token stream."""
        r = self.answer(message, language)
        for t in r.tool_calls:
            yield {"type": "tool", **t}
        for w in r.answer.split(" "):
            yield {"type": "token", "text": w + " "}
        yield {"type": "final", "result": r}


class PolicyIndex:
    """Lexical retrieval over the bilingual policy corpus.

    Score = weighted fraction of the query's content tokens found in a policy, where a
    token's weight is its inverse document frequency (rare words such as 'livraison'
    count more than words that appear in every policy) and title hits count double.
    """

    def __init__(self, path: Path):
        import math

        with open(path, encoding="utf-8") as fh:
            self.policies: List[dict] = json.load(fh)
        self._body = {
            p["id"]: set(tokens(" ".join([p["text_en"], p["text_fr"]]))) for p in self.policies
        }
        self._title = {
            p["id"]: set(tokens(" ".join([p["title_en"], p["title_fr"]]))) for p in self.policies
        }
        n = len(self.policies)
        df: Dict[str, int] = {}
        for pid in self._body:
            for t in self._body[pid] | self._title[pid]:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self._unknown_idf = math.log(1 + n)  # a query word found in no policy

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        q = set(tokens(query))
        if not q:
            return []
        total = sum(self._idf.get(t, self._unknown_idf) for t in q)
        scored: List[Dict[str, Any]] = []
        for p in self.policies:
            pid = p["id"]
            got = 0.0
            for t in q:
                w = self._idf.get(t, 0.0)
                if t in self._title[pid]:
                    got += 2 * w
                elif t in self._body[pid]:
                    got += w
            scored.append({"policy": p, "raw": got / total, "score": min(1.0, got / total)})
        scored.sort(key=lambda s: s["raw"], reverse=True)  # rank on the uncapped score so a title hit beats a body hit
        return scored[:top_k]


class RuleBaselineProvider(Provider):
    name = "rule"

    def __init__(self, policies: PolicyIndex, recommend: Callable[[List[str], int], List[dict]],
                 describe: Callable[[str], Optional[str]], retrieval_floor: float = 0.4):
        self.policies = policies
        self.recommend = recommend
        self.describe = describe
        self.floor = retrieval_floor

    def _refuse(self, reason: str, lang: str) -> ChatResult:
        return ChatResult(answer=MESSAGES[reason][lang], language=lang, provider=self.name,
                          refused=True, refusal_reason=reason)

    def answer(self, message: str, language: str) -> ChatResult:
        lang = language
        norm = strip_accents(message.lower())

        # Guardrail 1: never disclose another customer's data.
        if OTHER_CUSTOMER.search(norm):
            return self._refuse("privacy", lang)

        codes = PRODUCT_CODE.findall(message)
        words = set(tokens(message))
        wants_reco = any(w.startswith(RECO_STEMS) for w in words)
        wants_product = bool(words & {"product", "produit", "price", "prix", "item", "article"})
        wants_retention = bool(words & RETENTION_WORDS)

        # Tool 3: recommendations (the codes come from the Head B engine, never from text).
        if codes and wants_reco:
            codes = [c.upper() for c in codes]
            recs = self.recommend(codes, 5)
            lines = []
            for r in recs:
                tag = ""
                if r.get("because_of"):
                    tag = (" (parce que vous avez ajouté " if lang == "fr" else " (because you added ") + \
                          f"{r['because_of']})"
                lines.append(f"- {r['item']}: {r.get('description') or '?'}{tag}")
            head = "Suggestions :" if lang == "fr" else "Suggestions:"
            return ChatResult(
                answer=head + "\n" + "\n".join(lines), language=lang, provider=self.name,
                tool_calls=[{"tool": "recommendations", "arguments": {"basket": codes, "k": 5},
                             "result_summary": f"{len(recs)} items ({sum(r['strategy']=='rule' for r in recs)} by rule)"}],
            )

        # Tool 4: product lookup.
        if codes and (wants_product or not wants_reco):
            code = codes[0].upper()
            desc = self.describe(code)
            if desc:
                txt = f"{code} : {desc}." if lang == "fr" else f"{code}: {desc}."
                return ChatResult(answer=txt, language=lang, provider=self.name,
                                  tool_calls=[{"tool": "product_lookup", "arguments": {"code": code},
                                               "result_summary": desc}])
            return ChatResult(
                answer=("Je ne trouve pas ce code produit." if lang == "fr" else "I couldn't find that product code."),
                language=lang, provider=self.name,
                tool_calls=[{"tool": "product_lookup", "arguments": {"code": code}, "result_summary": "not found"}])

        # Tool 2: retention score needs customer features; the chat never invents a number.
        if wants_retention:
            return ChatResult(answer=MESSAGES["retention_tool"][lang], language=lang, provider=self.name,
                              tool_calls=[{"tool": "retention_info", "arguments": {},
                                           "result_summary": "redirected to the Prediction page (no probability in chat)"}])

        # Guardrail 2: out of scope (no domain vocabulary at all).
        if not _in_domain(message):
            return self._refuse("out_of_scope", lang)

        # Tool 1: policy search with a similarity floor and citations.
        hits = self.policies.search(message)
        if not hits or hits[0]["score"] < self.floor:
            return self._refuse("no_evidence", lang)
        best = hits[0]["policy"]
        text = best["text_fr"] if lang == "fr" else best["text_en"]
        title = best["title_fr"] if lang == "fr" else best["title_en"]
        return ChatResult(
            answer=text, language=lang, provider=self.name,
            citations=[{"source_id": best["id"], "title": title, "score": round(hits[0]["score"], 3)}],
            tool_calls=[{"tool": "policy_search", "arguments": {"query": message},
                         "result_summary": f"top hit {best['id']} ({hits[0]['score']:.2f})"}],
        )


class AssistantService:
    def __init__(self, providers: Dict[str, Provider]):
        self.providers = providers

    def chat(self, message: str, language: str = "auto", provider: str = "rule") -> ChatResult:
        lang = detect_language(message) if language == "auto" else language
        prov = self.providers.get(provider)
        if prov is None:
            return ChatResult(answer=MESSAGES["not_implemented"][lang], language=lang, provider=provider,
                              refused=True, refusal_reason="not_implemented")
        return prov.answer(message, lang)

    def stream(self, message: str, language: str = "auto", provider: str = "rule") -> Iterator[Dict[str, Any]]:
        lang = detect_language(message) if language == "auto" else language
        prov = self.providers.get(provider)
        if prov is None:
            r = ChatResult(answer=MESSAGES["not_implemented"][lang], language=lang, provider=provider,
                           refused=True, refusal_reason="not_implemented")
            yield {"type": "token", "text": r.answer}
            yield {"type": "final", "result": r}
            return
        yield from prov.stream(message, lang)
