"""Embedding retrieval over the bilingual policy corpus (replaces the lexical baseline).

Each policy is embedded once per language (title + text), so a French question can match the
English chunk of the same policy and vice versa. Score = cosine similarity of the best chunk.

Embedders:
* a real multilingual model through an LLMClient (OpenAI text-embedding-3-small, or Ollama bge-m3)
* HashingEmbedder: an OFFLINE fallback built on hashed word and character n-grams. It is
  deterministic and needs no network, but it is lexical (shared sub-words), NOT semantic, so it
  will not match paraphrases that share no letters. It exists so CI and the deployed demo work
  without an API key; the real embedders are what the spec calls for.
"""
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Protocol

from app.services.assistant import strip_accents, tokens


class Embedder(Protocol):
    name: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dims: int = 1024):
        self.dims = dims

    def _bucket(self, feat: str) -> int:
        return int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=4).digest(), "big") % self.dims

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dims
        for w in tokens(text):
            v[self._bucket("w:" + w)] += 2.0
            padded = f"^{strip_accents(w)}$"
            for n in (3, 4):
                for i in range(len(padded) - n + 1):
                    v[self._bucket(f"c{n}:" + padded[i:i + n])] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


def _normalise(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class EmbeddingIndex:
    """Drop-in replacement for PolicyIndex: same `.policies` and `.search()` shape."""

    def __init__(self, path: Path, embedder: Embedder):
        with open(path, encoding="utf-8") as fh:
            self.policies: List[dict] = json.load(fh)
        self.embedder = embedder
        chunks: List[str] = []
        owners: List[str] = []
        for p in self.policies:
            for lang in ("en", "fr"):
                chunks.append(f"{p['title_' + lang]}. {p['text_' + lang]}")
                owners.append(p["id"])
        self._owners = owners
        self._vecs = [_normalise(v) for v in embedder.embed(chunks)]

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        if not query.strip():
            return []
        q = _normalise(self.embedder.embed([query])[0])
        best: Dict[str, float] = {}
        for pid, vec in zip(self._owners, self._vecs, strict=True):
            s = sum(a * b for a, b in zip(q, vec, strict=True))
            best[pid] = max(best.get(pid, -1.0), s)
        by_id = {p["id"]: p for p in self.policies}
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{"policy": by_id[pid], "raw": s, "score": max(0.0, min(1.0, s))} for pid, s in ranked]
