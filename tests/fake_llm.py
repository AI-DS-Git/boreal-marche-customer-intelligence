"""A tiny local HTTP server that speaks the OpenAI-compatible and Ollama wire formats, so the real
clients (app/services/llm.py) are exercised over real sockets with no network and no API key.

A test supplies `script(messages, tools) -> list of steps`, where a step is
("token", text) | ("tool", name, args_dict) | ("sleep", seconds).
Embedding endpoints return the offline hashing embedder's vectors."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, List, Tuple

from app.services.retrieval import HashingEmbedder

Step = Tuple[Any, ...]
_EMB = HashingEmbedder()


class FakeLLM:
    def __init__(self, script: Callable[[List[dict], List[dict]], List[Step]]):
        self.script = script
        self.chat_calls = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n) or b"{}")

            def _send(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _line(self, text):
                self.wfile.write(text.encode())
                self.wfile.flush()

            def do_POST(self):
                body = self._body()
                if self.path.endswith("/embeddings"):
                    vecs = _EMB.embed(body["input"])
                    return self._send({"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]})
                if self.path.endswith("/api/embed"):
                    return self._send({"embeddings": _EMB.embed(body["input"])})
                outer.chat_calls += 1
                steps = outer.script(body["messages"], body.get("tools") or [])
                self.send_response(200)
                openai = self.path.endswith("/chat/completions")
                self.send_header("Content-Type", "text/event-stream" if openai else "application/x-ndjson")
                self.end_headers()
                idx = 0
                for st in steps:
                    if st[0] == "sleep":
                        time.sleep(st[1])
                    elif st[0] == "token":
                        if openai:
                            self._line("data: " + json.dumps({"choices": [{"delta": {"content": st[1]}}]}) + "\n\n")
                        else:
                            self._line(json.dumps({"message": {"role": "assistant", "content": st[1]},
                                                   "done": False}) + "\n")
                    elif st[0] == "tool":
                        if openai:
                            args = json.dumps(st[2])
                            half = len(args) // 2  # arguments arrive in fragments, like the real API
                            for frag, first in ((args[:half], True), (args[half:], False)):
                                tc = {"index": idx, "function": {"arguments": frag}}
                                if first:
                                    tc["id"] = f"call_{idx}"
                                    tc["function"]["name"] = st[1]
                                self._line("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [tc]}}]}) + "\n\n")
                            idx += 1
                        else:
                            self._line(json.dumps({"message": {"role": "assistant", "content": "", "tool_calls": [
                                {"function": {"name": st[1], "arguments": st[2]}}]}, "done": False}) + "\n")
                self._line("data: [DONE]\n\n" if openai else json.dumps({"done": True}) + "\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    @property
    def openai_base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def ollama_base(self) -> str:
        return self.openai_base
