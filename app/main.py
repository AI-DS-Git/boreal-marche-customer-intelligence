"""Boréal Marché Customer Intelligence API (FastAPI).

Thin HTTP layer: validation (Pydantic v2, extra="forbid"), model/rules/assistant services,
structured logging to the prediction log, and monitoring endpoints. All business logic
lives in app/services so it can be unit-tested without a web server.
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import sklearn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.db import LogStore
from app.monitoring import MonitoringService
from app.schemas import (
    BatchPredictRequest, BatchPredictResponse, ChatRequest, ChatResponse, Citation,
    ExplainRequest, ExplainResponse, HealthResponse, PredictRequest, PredictResponse, ReadyResponse, RecommendRequest,
    RecommendResponse, ToolCall, VersionResponse,
)
from app.services.assistant_factory import build_assistant
from app.services.model_service import ModelService
from app.services.rules_service import RulesService

log = logging.getLogger("boreal.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def jlog(**kw):
    log.info(json.dumps(kw, default=str))


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.settings = s
    # Loaded once at start-up, not per request. A missing/corrupt artifact fails the
    # container here (and /ready never turns green) instead of failing user requests.
    app.state.model = ModelService(s.model_path, s.model_metadata_path, s.strict_library_versions,
                                       reference_path=s.drift_reference_path)
    if app.state.model.version_warning:
        jlog(event="model_version_warning", detail=app.state.model.version_warning)
    app.state.rules = RulesService(s.rules_path, s.popularity_path, s.products_path)
    app.state.assistant = build_assistant(s, app.state.rules)
    app.state.store = LogStore(s.database_url)
    purged = app.state.store.purge_older_than(s.log_retention_days)  # retention policy, applied at every start-up
    reference = None
    if s.drift_reference_path.exists():
        reference = json.loads(s.drift_reference_path.read_text(encoding="utf-8"))
    app.state.monitor = MonitoringService(app.state.store, reference, s)
    app.state.metrics = {"requests": 0, "errors": 0}
    jlog(event="startup", git_sha=s.git_sha, model_version=app.state.model.model_version, purged=purged)
    yield


app = FastAPI(title="Boréal Marché Customer Intelligence API", version="0.9.0", lifespan=lifespan)


# ---- cross-cutting: request id, timing, request log, uniform errors ----------------------
@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    request.state.meta = {}
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = rid
        return response
    finally:
        ms = (time.perf_counter() - t0) * 1000
        path = request.url.path
        store = getattr(request.app.state, "store", None)
        if store is not None and path not in ("/health", "/ready", "/metrics"):
            try:
                store.log_request(rid, path, status, ms, request.state.meta or None)
            except Exception as exc:  # logging must never take the API down
                jlog(event="log_failure", error=str(exc))
        m = getattr(request.app.state, "metrics", None)
        if m is not None:
            m["requests"] += 1
            m["errors"] += int(status >= 500)
        jlog(event="request", request_id=rid, path=path, status=status, latency_ms=round(ms, 2))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errs = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "request_id": getattr(request.state, "request_id", None), "error": "validation_error",
        "detail": json.dumps(errs)})


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={
        "request_id": getattr(request.state, "request_id", None), "error": "http_error", "detail": str(exc.detail)})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    jlog(event="unhandled", request_id=getattr(request.state, "request_id", None), error=repr(exc))
    return JSONResponse(status_code=500, content={
        "request_id": getattr(request.state, "request_id", None), "error": "internal_error",
        "detail": "Unexpected error; see logs with this request_id."})


# ---- health / version -----------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness: the process is up. Does not touch dependencies."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request):
    """Readiness: model loaded, rules loaded, database reachable."""
    st = request.app.state
    checks = {"model": hasattr(st, "model"), "rules": hasattr(st, "rules"),
              "database": bool(getattr(st, "store", None) and st.store.ping())}
    ok = all(checks.values())
    body = ReadyResponse(status="ready" if ok else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())


@app.get("/version", response_model=VersionResponse)
def version(request: Request):
    st = request.app.state
    return VersionResponse(
        git_sha=st.settings.git_sha, model_version=st.model.model_version,
        model_git_commit=st.model.metadata.get("git_commit"), decision_threshold=st.model.threshold,
        rules_count=len(st.rules.rules), sklearn_version=sklearn.__version__)


# ---- Head A: retention ----------------------------------------------------------------
def _to_response(rid, model, features, p_retain, latency_ms) -> PredictResponse:
    return PredictResponse(
        request_id=rid, retain_probability=round(float(p_retain), 6),
        churn_risk=round(1.0 - float(p_retain), 6), decision=model.decide(float(p_retain)),
        threshold=model.threshold, model_version=model.model_version, latency_ms=round(latency_ms, 2))


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    st = request.app.state
    t0 = time.perf_counter()
    p = float(st.model.predict_proba([body.features])[0])
    ms = (time.perf_counter() - t0) * 1000
    resp = _to_response(request.state.request_id, st.model, body.features, p, ms)
    st.store.log_prediction(resp.request_id, "/predict", st.model.model_version, st.settings.git_sha,
                            json.loads(body.features.model_dump_json()), p, resp.decision, ms)
    return resp


@app.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest, request: Request):
    """Top factors behind one prediction (occlusion vs a typical customer). Staff-facing: it is not logged as a
    prediction (so it does not feed drift statistics) and it is not a customer-ready 'reason'."""
    st = request.app.state
    p = float(st.model.predict_proba([body.features])[0])
    return ExplainResponse(
        request_id=request.state.request_id, retain_probability=round(p, 6),
        factors=st.model.explain(body.features, body.k),
        method="group occlusion: each set of related predictors replaced by typical-customer values, one group at a time",
        caveat="Effects are in points of the uncalibrated model score. They describe how the model reacts, not why the customer behaves this way.")


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest, request: Request):
    st = request.app.state
    t0 = time.perf_counter()
    probs = st.model.predict_proba(body.items)
    ms = (time.perf_counter() - t0) * 1000
    per_item = ms / len(body.items)
    rid = request.state.request_id
    out = []
    for i, (f, p) in enumerate(zip(body.items, probs, strict=True)):
        r = _to_response(f"{rid}-{i}", st.model, f, p, per_item)
        st.store.log_prediction(r.request_id, "/predict/batch", st.model.model_version, st.settings.git_sha,
                                json.loads(f.model_dump_json()), float(p), r.decision, per_item)
        out.append(r)
    return BatchPredictResponse(request_id=rid, predictions=out, count=len(out))


# ---- Head B: recommendations ----------------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest, request: Request):
    t0 = time.perf_counter()
    recs = request.app.state.rules.recommend(body.basket, body.k)
    request.state.meta = {"rule_hits": sum(1 for r in recs if r.get("strategy") == "rule"), "k": body.k}
    return RecommendResponse(request_id=request.state.request_id, recommendations=recs,
                             latency_ms=round((time.perf_counter() - t0) * 1000, 2))


# ---- Head C: assistant ----------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request):
    t0 = time.perf_counter()
    r = request.app.state.assistant.chat(body.message, body.language, body.provider)
    request.state.meta = {"provider": r.provider, "refused": r.refused, "reason": r.refusal_reason}
    return ChatResponse(
        request_id=request.state.request_id, answer=r.answer, language=r.language, provider=r.provider,
        citations=[Citation(**c) for c in r.citations], tool_calls=[ToolCall(**t) for t in r.tool_calls],
        refused=r.refused, refusal_reason=r.refusal_reason, latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post("/chat/stream")
def chat_stream(body: ChatRequest, request: Request):
    """Server-sent events: tool / token / replace events as they happen, then one final event
    carrying the same fields as /chat. 'replace' tells the client to discard the text streamed so
    far (a guardrail or the answer verifier overrode the model)."""
    t0 = time.perf_counter()
    rid = request.state.request_id
    request.state.meta = {"provider": body.provider, "stream": True}
    svc = request.app.state.assistant

    def events():
        for ev in svc.stream(body.message, body.language, body.provider):
            if ev["type"] == "final":
                r = asdict(ev["result"])
                ev = {"type": "final", "request_id": rid, **r,
                      "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- monitoring -----------------------------------------------------------------------
@app.get("/monitoring/summary")
def monitoring_summary(request: Request, hours: float = 24):
    mon = request.app.state.monitor
    summary = mon.summary(hours)
    summary["new_alerts"] = mon.raise_alerts(summary)
    if summary["new_alerts"]:  # show an alert raised by THIS request in the same response
        summary["alerts"] = mon.store.recent_alerts(10)
    return summary


@app.post("/monitoring/outcomes")
def record_outcome(payload: dict, request: Request):
    """Backfill the delayed ground truth: {"request_id": "...", "observed_retained": 0|1}."""
    rid, y = payload.get("request_id"), payload.get("observed_retained")
    if not isinstance(rid, str) or y not in (0, 1):
        return JSONResponse(status_code=422, content={"error": "validation_error",
                            "detail": "need request_id (str) and observed_retained (0 or 1)"})
    request.app.state.store.record_outcome(rid, int(y))
    return {"status": "recorded", "request_id": rid}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request):
    """Prometheus-style plain-text counters."""
    m = request.app.state.metrics
    return (f"boreal_requests_total {m['requests']}\n"
            f"boreal_errors_total {m['errors']}\n")
