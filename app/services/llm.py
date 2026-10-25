"""Minimal HTTP clients for the two LLM back-ends behind one interface (spec 3.4).

Only the standard library is used (urllib), so there is no SDK to pin and the clients can be
tested against a local fake server. Two back-ends:

* OpenAI-compatible (hosted):  POST {base}/chat/completions (SSE stream), POST {base}/embeddings
* Ollama (local):              POST {base}/api/chat (NDJSON stream),      POST {base}/api/embed

`chat_stream` yields events:
    {"type": "token", "text": str}
    {"type": "tool_call", "id": str, "name": str, "arguments": dict}
    {"type": "done"}
API keys are only ever sent in the Authorization header and are never logged.
"""
import json
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

Event = Dict[str, Any]
Message = Dict[str, Any]


class LLMError(RuntimeError):
    """The back-end was unreachable, timed out, or returned something unusable."""


def _check_url(url: str) -> str:
    if urlparse(url).scheme not in ("http", "https"):
        raise LLMError("only http(s) URLs are allowed")
    return url


def _open(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float):
    req = urllib.request.Request(  # noqa: S310 - scheme checked above
        _check_url(url), data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code} from {urlparse(url).netloc}") from e
    except (urllib.error.URLError, socket.timeout, OSError) as e:
        raise LLMError(f"cannot reach {urlparse(url).netloc}: {type(e).__name__}") from e


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    with _open(url, payload, headers, timeout) as resp:
        try:
            return json.loads(resp.read().decode("utf-8"))
        except (ValueError, OSError) as e:
            raise LLMError("invalid JSON from back-end") from e


def _post_lines(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Iterator[str]:
    """Yield decoded lines as they arrive (this is what makes the streaming real)."""
    with _open(url, payload, headers, timeout) as resp:
        try:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line
        except (OSError, UnicodeDecodeError) as e:
            raise LLMError("stream interrupted") from e


class LLMClient(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]: ...

    @abstractmethod
    def chat_stream(self, messages: List[Message], tools: List[Dict[str, Any]]) -> Iterator[Event]: ...

    @abstractmethod
    def assistant_tool_message(self, calls: List[Event]) -> Message:
        """The assistant turn that requested `calls`, in this back-end's wire format."""

    @abstractmethod
    def tool_message(self, call: Event, content: str) -> Message:
        """The tool-result turn answering `call`."""


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, api_key: str, chat_model: str = "gpt-4o-mini",
                 embed_model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1", timeout: float = 30.0):
        self.chat_model, self.embed_model = chat_model, embed_model
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts: List[str]) -> List[List[float]]:
        data = _post_json(f"{self.base}/embeddings", {"model": self.embed_model, "input": texts},
                          self._headers, self.timeout)
        try:
            rows = sorted(data["data"], key=lambda r: r["index"])
            return [r["embedding"] for r in rows]
        except (KeyError, TypeError) as e:
            raise LLMError("unexpected embeddings response") from e

    def chat_stream(self, messages: List[Message], tools: List[Dict[str, Any]]) -> Iterator[Event]:
        payload: Dict[str, Any] = {"model": self.chat_model, "messages": messages, "stream": True,
                                   "temperature": 0}
        if tools:
            payload["tools"] = tools
        pending: Dict[int, Dict[str, Any]] = {}
        for line in _post_lines(f"{self.base}/chat/completions", payload, self._headers, self.timeout):
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0]["delta"]
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise LLMError("unexpected stream chunk") from e
            if delta.get("content"):
                yield {"type": "token", "text": delta["content"]}
            for tc in delta.get("tool_calls") or []:
                slot = pending.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["name"] = fn.get("name") or slot["name"]
                slot["args"] += fn.get("arguments") or ""
        for i in sorted(pending):
            slot = pending[i]
            try:
                args = json.loads(slot["args"]) if slot["args"] else {}
            except ValueError:
                args = {}
            yield {"type": "tool_call", "id": slot["id"] or f"call_{i}", "name": slot["name"], "arguments": args}
        yield {"type": "done"}

    def assistant_tool_message(self, calls: List[Event]) -> Message:
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}

    def tool_message(self, call: Event, content: str) -> Message:
        return {"role": "tool", "tool_call_id": call["id"], "content": content}


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", chat_model: str = "llama3.1:8b",
                 embed_model: str = "bge-m3", timeout: float = 120.0):
        self.base, self.chat_model, self.embed_model = base_url.rstrip("/"), chat_model, embed_model
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        data = _post_json(f"{self.base}/api/embed", {"model": self.embed_model, "input": texts}, {}, self.timeout)
        try:
            return [list(map(float, row)) for row in data["embeddings"]]
        except (KeyError, TypeError, ValueError) as e:
            raise LLMError("unexpected embed response") from e

    def chat_stream(self, messages: List[Message], tools: List[Dict[str, Any]]) -> Iterator[Event]:
        payload: Dict[str, Any] = {"model": self.chat_model, "messages": messages, "stream": True,
                                   "options": {"temperature": 0}}
        if tools:
            payload["tools"] = tools
        n = 0
        for line in _post_lines(f"{self.base}/api/chat", payload, {}, self.timeout):
            try:
                chunk = json.loads(line)
            except ValueError as e:
                raise LLMError("unexpected stream chunk") from e
            msg = chunk.get("message") or {}
            if msg.get("content"):
                yield {"type": "token", "text": msg["content"]}
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                n += 1
                yield {"type": "tool_call", "id": f"call_{n}", "name": fn.get("name", ""), "arguments": args}
            if chunk.get("done"):
                break
        yield {"type": "done"}

    def assistant_tool_message(self, calls: List[Event]) -> Message:
        return {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}

    def tool_message(self, call: Event, content: str) -> Message:
        return {"role": "tool", "tool_name": call["name"], "content": content}


def build_client(kind: str, *, openai_key: Optional[str], openai_base: str, openai_chat: str,
                 openai_embed: str, ollama_base: Optional[str], ollama_chat: str,
                 ollama_embed: str) -> Optional[LLMClient]:
    """Returns None when the back-end is not configured (no key / no base URL)."""
    if kind == "openai" and openai_key:
        return OpenAIClient(openai_key, openai_chat, openai_embed, openai_base)
    if kind == "ollama" and ollama_base:
        return OllamaClient(ollama_base, ollama_chat, ollama_embed)
    return None
