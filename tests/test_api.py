"""Contract tests against the real FastAPI app. Skipped where FastAPI is not installed
(the CI image and the Docker image both have it)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from tests.helpers import golden  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["DATABASE_URL"] = "sqlite:///" + str(tmp_path_factory.mktemp("db") / "t.db")
    with TestClient(app) as c:
        yield c


def test_health_ready_version(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"
    v = client.get("/version").json()
    assert v["decision_threshold"] == 0.82 and v["rules_count"] > 0


def test_predict_golden_customer(client):
    g = golden()
    r = client.post("/predict", json={"features": g["features"]})
    assert r.status_code == 200
    b = r.json()
    assert abs(b["retain_probability"] - g["expected_retain_probability"]) < 1e-5
    assert b["decision"] == g["expected_decision"]
    assert abs(b["retain_probability"] + b["churn_risk"] - 1) < 1e-5
    assert r.headers["x-request-id"] == b["request_id"]


def test_predict_rejects_bad_input_with_uniform_error(client):
    f = dict(golden()["features"], surprise=1)
    r = client.post("/predict", json={"features": f})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_batch_and_recommend_and_chat(client):
    f = golden()["features"]
    assert client.post("/predict/batch", json={"items": [f, f]}).json()["count"] == 2
    rec = client.post("/recommend", json={"basket": ["85123A"], "k": 3}).json()
    assert 1 <= len(rec["recommendations"]) <= 3
    chat = client.post("/chat", json={"message": "What is your return policy?"}).json()
    assert chat["citations"] and not chat["refused"]


def test_monitoring_summary_and_metrics(client):
    s = client.get("/monitoring/summary").json()
    assert s["overall_status"] in ("green", "amber", "red", "insufficient_data")
    assert "first_true_accuracy_signal_on" in s["ground_truth"]
    assert "boreal_requests_total" in client.get("/metrics").text


def sse_events(text):
    import json as _json
    return [_json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_chat_stream_emits_tokens_then_one_final_event(client):
    r = client.post("/chat/stream", json={"message": "What is your return policy?", "provider": "rule"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = sse_events(r.text)
    assert [e["type"] for e in ev].count("final") == 1 and ev[-1]["type"] == "final"
    assert any(e["type"] == "token" for e in ev)
    final = ev[-1]
    assert final["citations"][0]["source_id"] == "returns" and not final["refused"]
    assert "".join(e["text"] for e in ev if e["type"] == "token").strip() == final["answer"].strip()


def test_chat_stream_unconfigured_provider_is_reported(client):
    r = client.post("/chat/stream", json={"message": "hello", "provider": "openai"})
    final = sse_events(r.text)[-1]
    assert final["refused"] and final["refusal_reason"] == "not_implemented"
