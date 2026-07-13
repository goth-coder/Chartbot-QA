"""Flask backend for Chart-Visual-QA.

Mock-first: the API contract is stable and returns fake answers via
``inference.run_inference`` until the real model lands. See docs/PLAN.md §9.

Endpoints:
    GET  /api/health     -> {"status": "ok", "mock": <bool>}
    POST /api/ask        -> {"answer": <str>, "mock": <bool>, "is_chart": <bool>,
                             "latency_ms": <number>}
                            or, if the guard blocks the question (HTTP 200):
                            {"blocked": true, "category": <str>, "reason": <str>}
                            (multipart/form-data: image=<file>, question=<string>)
    POST /api/ask/stream -> same body/contract, same pipeline, but as
                            text/event-stream: one ``data: {...}`` line per pipeline
                            stage ({"stage": <name>, "status": "start"|"done",
                            "elapsed_ms": <number|null>}), ending with
                            {"stage": "result", "body": <the /api/ask JSON above>,
                            "status_code": <int>}. Powers the frontend's per-stage
                            progress UI; both endpoints share the same pipeline
                            generator (_ask_events) so they can never diverge.

``is_chart`` is a cheap Layer-1 heuristic (see chart_check) — a warning signal, not
a hard block: when false, the UI can warn that results may be unreliable.
The Layer-2 guard (see guard.py) screens the question for toxicity / prompt
injection / PII before the model runs.

Run standalone:   python backend/app.py
Or via Flask CLI: flask --app backend/app run --port 5000
Or via the root orchestrator:  python app.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load <repo-root>/.env BEFORE importing modules that read env at import time
# (inference.USE_MOCK, guard.* thresholds, guard_llm.GUARD_LLM_*).
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

import answer_cache
import auth
import budget
import conversation_store
import feedback_store
import metrics
import ratelimit
import vlm_provider
from chart_check import looks_like_chart
from env_config import env_bool, env_float, env_int, env_str
from guard import guard, warmup
from inference import is_mock, run_inference
from uploads import InvalidImage, sanitize_image

logging.basicConfig(level=logging.INFO)

# Demo toggle: when MOCK_REVEAL is on, mock mode returns the canned answer instead
# of the disclaimer — useful for demoing the full UI. Off keeps Rule 3 (no fake
# numbers, disclaimer only).
MOCK_REVEAL = env_bool("MOCK_REVEAL")

app = Flask(__name__)

# Shown instead of a (fake) answer while the backend is in mock mode, so a
# canned value is never mistaken for a real model prediction.
MOCK_DISCLAIMER = (
    "Mock mode: no model is connected yet, so no answer is produced. "
    "This is a placeholder response for building and testing the app."
)

# CORS: pin to the frontend origin(s) in prod via CORS_ORIGINS (comma-separated); '*'
# is allowed for dev/local tools. In dev the Vite proxy makes requests same-origin anyway.
_CORS_ORIGINS = [o.strip() for o in env_str("CORS_ORIGINS").split(",") if o.strip()] or ["*"]
CORS(app, resources={r"/api/*": {"origins": _CORS_ORIGINS}})


@app.after_request
def _security_headers(resp):
    """Baseline response hardening (defense-in-depth; the prod proxy may add more)."""
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.path.startswith("/api/"):
        metrics.count_request(request.path, resp.status_code)
    return resp

# Cap uploads (MB) so a huge file can't exhaust memory. Keep in sync with the
# frontend's MAX_BYTES.
MAX_UPLOAD_MB = env_float("MAX_UPLOAD_MB")
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)


# Min "meaningful" (alphanumeric) chars a question must have to not be junk.
MIN_QUESTION_ALNUM = env_int("MIN_QUESTION_ALNUM")

# Below this, the chart gate is confident enough to hard-block the VLM call outright
# (not just warn) — see chart_check.py / .env.example for why.
CHART_BLOCK_THRESHOLD = env_float("CHART_BLOCK_THRESHOLD")


def _question_too_weak(question: str) -> bool:
    """Reject only near-empty / junk questions (e.g. "?", "hi").

    Counts "meaningful" characters — letters or digits in ANY language, so CJK
    questions (which have no spaces) and short questions are handled the same way.
    This is a light Layer-1 guard against junk, not real NLP.
    """
    meaningful = sum(1 for c in question if c.isalnum())
    return meaningful < MIN_QUESTION_ALNUM


def _client_ip() -> str:
    """Best-effort client IP for rate limiting. Behind nginx / Cloud Run the real client
    is the left-most entry of X-Forwarded-For; fall back to the socket peer.

    NOTE: the left-most XFF entry is client-controlled, so the per-IP rate limit is
    **best-effort** — a determined bot can rotate a forged X-Forwarded-For to dodge it.
    That's acceptable here because the hard GPU-cost backstop is the IP-independent daily
    **budget breaker** (budget.py): a spoofer still hits the VLM_DAILY_BUDGET 429 wall.
    The limiter is defense-in-depth against casual/accidental floods, not the cost cap."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _authenticated_user() -> dict | None:
    """Verify the request's Google ID token (Authorization: Bearer <token>), or None.

    Always returns None when AUTH_ENABLED=0 (local/--dev/tests) — callers that require
    a signed-in user do so via ``_require_auth()`` below, which is the actual gate;
    this helper just answers "who, if anyone" for both that gate and the rate limiter.
    """
    if not auth.AUTH_ENABLED:
        return None
    header = request.headers.get("Authorization", "")
    token = header[7:] if header.startswith("Bearer ") else ""
    return auth.verify_google_token(token)


def _require_auth():
    """Return a ready-to-return ``(body, 401)`` if sign-in is required and
    missing/invalid, else None (caller does ``if (r := _require_auth()): return r``)."""
    if not auth.AUTH_ENABLED:
        return None
    if _authenticated_user() is None:
        return jsonify(error="Sign in with Google to use this."), 401
    return None


@app.get("/api/health")
def health():
    return jsonify(status="ok", mock=is_mock())


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus scrape target (fail-open: a short notice if prometheus_client is absent)."""
    body, content_type = metrics.render()
    return Response(body, mimetype=content_type)


@app.get("/api/vlm/warm")
def vlm_warm():
    """Fire-and-forget nudge for the remote VLM (see vlm_provider.py). The frontend
    calls this right after sign-in so a cold RunPod dev pod / Cloud Run instance has a
    head start before the user's first real question. Always returns immediately.

    Gated the same as /api/ask (Phase 3.7: required sign-in) — no reason to warm the
    billed GPU for a visitor who hasn't authenticated."""
    if (resp := _require_auth()) is not None:
        return resp
    if not is_mock():
        vlm_provider.warm()
    return jsonify(status="ok")


@app.get("/api/guard/warm")
def guard_warm():
    """Fire-and-forget nudge for the remote Layer-3 guard (Llama Guard on Ollama, a
    separate scale-to-zero Cloud Run service). The frontend calls this alongside
    /api/vlm/warm right after sign-in / page load, so the guard's cold start
    (container spin-up + ~15-18s model load) happens during the user's think-time
    instead of blocking their first question by ~90s.

    guard_llm.warmup() blocks for the whole cold load, so run it in a daemon thread and
    return immediately. No-op in mock mode and when the guard LLM is disabled/unavailable
    (warmup() itself guards those). Gated like /api/ask — no warming for anonymous visitors."""
    if (resp := _require_auth()) is not None:
        return resp
    if not is_mock():
        threading.Thread(target=_guard_warm_bg, daemon=True).start()
    return jsonify(status="ok")


def _guard_warm_bg() -> None:
    """Warm only the remote Layer-3 LLM (the ~90s cold-start surface). The in-process
    Layer-2 encoders are already warmed at boot (gunicorn.conf.py post_worker_init), so
    this deliberately does NOT call guard.warmup() — just the Ollama round-trip."""
    try:
        import guard_llm
        guard_llm.warmup()
    except Exception:  # noqa: BLE001
        pass


@app.post("/api/feedback")
def feedback():
    """Record a 👍/👎 on an assistant answer (Phase 5 data flywheel). Re-hydrates the
    chart image + the answer being rated from the conversation store and hands them to
    feedback_store (fire-and-forget → GCS). Gated like /api/ask.

    Body (JSON or form): ``conversation_id`` (required), ``vote`` in {"up","down"}
    (required), optional ``note``. Returns 200 immediately — persistence is best-effort;
    the vote is telemetry, not something whose durability the user should wait on."""
    if (resp := _require_auth()) is not None:
        return resp

    data = request.get_json(silent=True) or request.form
    conversation_id = (data.get("conversation_id") or "").strip()
    vote = (data.get("vote") or "").strip().lower()
    note = (data.get("note") or "").strip()

    if vote not in ("up", "down"):
        return jsonify(error="vote must be 'up' or 'down'."), 400
    if not conversation_id:
        return jsonify(error="conversation_id is required."), 400

    state = conversation_store.get(conversation_id)
    image_bytes = conversation_store.get_image(conversation_id)
    if state is None or image_bytes is None:
        # The conversation expired or never existed — nothing to attach the vote to.
        return jsonify(error="Unknown or expired conversation."), 404

    messages = state.get("messages", [])
    last_question = next((m["text"] for m in reversed(messages) if m["role"] == "user"), "")
    last_answer = next((m["text"] for m in reversed(messages) if m["role"] == "assistant"), "")

    feedback_store.record(
        conversation_id=conversation_id,
        question=last_question,
        model_answer=last_answer,
        vote=vote,
        note=note,
        image_bytes=image_bytes,
    )
    return jsonify(status="ok")


def _prepare_ask():
    """Auth + Layer-1 validation shared by /api/ask and /api/ask/stream — identical
    checks, run once, so the two endpoints can never drift apart on what they accept.

    Multi-turn (Phase 5): an optional ``conversation_id`` form field continues an existing
    chat. On turn 1 (no id) an image is required and a fresh id is minted; on a follow-up
    (valid id) the image is OPTIONAL — it's re-hydrated from the conversation store — but
    the client may still re-send it (e.g. if the store expired). The question is always
    required and screened by the guard on every turn.

    Returns ``(prep, None)`` on success where ``prep`` is a dict
    ``{question, image_bytes, rate_key, conversation_id, history, is_followup}``, or
    ``(None, (body, status))`` with a ready-to-return Flask response on the first failure.
    """
    if (resp := _require_auth()) is not None:
        return None, resp

    # Rate limit key: the signed-in user when auth is on (more precise than an IP a
    # bot can spoof via X-Forwarded-For), else client IP (AUTH_ENABLED=0).
    user = _authenticated_user()
    rate_key = user["sub"] if user else _client_ip()

    question = (request.form.get("question") or "").strip()
    image = request.files.get("image")
    conversation_id = (request.form.get("conversation_id") or "").strip()

    # --- Layer-1 guard: cheap rules, no ML (see docs/PLAN.md §6) ---
    if not question:
        return None, (jsonify(error="Please type a question."), 400)
    if _question_too_weak(question):
        return None, (jsonify(error="Please ask a more specific question."), 400)

    # Resolve the conversation: a valid existing id makes this a follow-up whose image can
    # come from the store; anything else starts a fresh conversation needing an image.
    history: list = []
    is_followup = False
    stored_state = conversation_store.get(conversation_id) if conversation_id else None
    if stored_state is not None:
        is_followup = True
        history = stored_state.get("messages", [])

    image_bytes = image.read() if (image is not None and image.filename != "") else b""

    if not image_bytes:
        if is_followup:
            # Follow-up with no re-uploaded image: pull the pinned chart from the store.
            image_bytes = conversation_store.get_image(conversation_id) or b""
        if not image_bytes:
            # Turn 1, or a follow-up whose stored image expired and wasn't re-sent.
            return None, (jsonify(error="Please upload an image."), 400)

    if not is_followup:
        conversation_id = conversation_store.new_id()

    prep = {
        "question": question,
        "image_bytes": image_bytes,
        "rate_key": rate_key,
        "conversation_id": conversation_id,
        "history": history,
        "is_followup": is_followup,
    }
    return prep, None


def _timed(fn, *args):
    """Run fn(*args), return (result, elapsed_seconds) — the caller's OWN wall-clock
    duration, independent of when/in-what-order the caller gets around to reading it.
    Needed because concurrent.futures' `future.result()` blocks the calling thread, so
    naively timing "before submit" to "after .result()" on a SECOND future would
    include however long the FIRST future's .result() call blocked for too."""
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def _ask_events(prep: dict):
    """The whole /api/ask pipeline, as a generator of progress events.

    ``prep`` is the dict from :func:`_prepare_ask`
    (``question, image_bytes, rate_key, conversation_id, history, is_followup``).

    Yields ``{"stage": <name>, "status": "start"|"done", "elapsed_ms": <float|None>}``
    for each stage, ending with exactly one
    ``{"stage": "result", "body": <dict>, "status_code": <int>}``. Shared by /api/ask
    (drains this, returns only the final event's body) and /api/ask/stream (emits
    every event as SSE) so the two can never diverge in behavior.

    Guard (question) and the chart gate (image) are independent — they run
    CONCURRENTLY via a thread pool. On this backend's single-vCPU Cloud Run
    allocation this doesn't buy true CPU parallelism, but it does overlap each
    stage's I/O waits (the Layer-3 guard's HTTP round-trip, the chart gate's
    Tesseract OCR subprocess) instead of paying for them back-to-back.

    Multi-turn (Phase 5): the guard runs on EVERY turn's question; the VLM receives the
    prior ``history`` so follow-ups are grounded in the conversation; after answering,
    the turn is appended to the conversation store and ``conversation_id`` is returned in
    the result so the client can send it on the next turn.
    """
    question = prep["question"]
    image_bytes = prep["image_bytes"]
    rate_key = prep["rate_key"]
    conversation_id = prep["conversation_id"]
    history = prep["history"]
    is_followup = prep["is_followup"]

    if not ratelimit.allow(rate_key):
        metrics.count_rate_limited()
        yield {"stage": "result",
               "body": {"error": "Too many requests — please slow down and try again shortly."},
               "status_code": 429}
        return

    # Re-encode the upload from its decoded pixels: rejects non-images and strips any
    # embedded/trailing payload, and yields canonical bytes for the cache key.
    try:
        image_bytes = sanitize_image(image_bytes)
    except InvalidImage:
        yield {"stage": "result", "body": {"error": "Uploaded file is not a valid image."},
               "status_code": 400}
        return

    # Answer cache (real mode only): a repeat (image, question) short-circuits the guard,
    # chart gate and VLM entirely. Never caches the mock disclaimer. Skipped for follow-up
    # turns: the answer depends on the conversation history, so an (image, question) key
    # alone would wrongly collide two follow-ups that share a question but differ in context.
    if not is_mock() and not is_followup:
        cached = answer_cache.get(image_bytes, question)
        metrics.count_cache(cached is not None)
        if cached is not None:
            # Still seed the conversation on a turn-1 cache hit so follow-ups have both the
            # pinned image and this first exchange in their history.
            conversation_store.start(conversation_id, image_bytes)
            conversation_store.append_turn(conversation_id, question, cached.get("answer", ""))
            yield {"stage": "result",
                   "body": {"mock": False, "cached": True, "latency_ms": 0.0,
                            "conversation_id": conversation_id, **cached},
                   "status_code": 200}
            return

    # --- Layer-2/3 guard (question) + Rule 4 chart gate (image) — run concurrently ---
    # as_completed() (not future.result() in submission order) so each stage's "done"
    # event fires the moment IT finishes, not after whichever we happen to check first
    # — a naive `guard_future.result()` then `chart_future.result()` would make the
    # SECOND-checked stage's measured elapsed_ms include the first stage's wait time
    # too, even if the second stage actually finished first (caught via /metrics
    # showing guard and chart_gate with near-identical, both-inflated sums, 2026-07-10).
    yield {"stage": "guard", "status": "start", "elapsed_ms": None}
    yield {"stage": "chart_gate", "status": "start", "elapsed_ms": None}
    stage_results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_timed, guard, question): "guard",
            pool.submit(_timed, looks_like_chart, image_bytes): "chart_gate",
        }
        for future in as_completed(futures):
            stage = futures[future]
            value, elapsed = future.result()
            stage_results[stage] = value
            metrics.observe_stage(stage, elapsed)
            yield {"stage": stage, "status": "done", "elapsed_ms": round(elapsed * 1000, 1)}
    verdict = stage_results["guard"]
    is_chart, chart_confidence = stage_results["chart_gate"]

    if not verdict.allowed:
        metrics.count_blocked(verdict.category)
        yield {"stage": "result",
               "body": {"blocked": True, "category": verdict.category, "reason": verdict.reason},
               "status_code": 200}
        return

    # Hard block when the gate is confident this ISN'T a chart — cheaper than the guard
    # LLM and MUCH cheaper than the VLM, and the only content check on the IMAGE itself
    # (guard() above only screens the question text, so an off-topic image with an
    # innocuous question would otherwise reach the model — confirmed 2026-07-10: an
    # unrelated photo got a bizarre off-topic answer instead of "not a chart"). A
    # borderline image (below CHART_CLIP_THRESHOLD but above this) still proceeds with
    # just the "may be unreliable" warning, so ambiguous-but-real charts aren't blocked.
    if chart_confidence < CHART_BLOCK_THRESHOLD:
        metrics.count_blocked("not_a_chart")
        yield {"stage": "result",
               "body": {"blocked": True, "category": "not_a_chart",
                        "reason": "This doesn't look like a chart — please upload a "
                                  "chart image."},
               "status_code": 200}
        return

    if not is_mock():
        # Daily VLM budget breaker (Phase 3.7): refuse *before* touching the GPU once the
        # day's budget is spent — the cache/guard above still work, the bill doesn't grow.
        if budget.over_budget():
            metrics.count_over_budget()
            yield {"stage": "result",
                   "body": {"error": "The demo's daily model quota has been reached — "
                                     "please try again tomorrow."},
                   "status_code": 429}
            return
        yield {"stage": "vlm_start", "status": "start", "elapsed_ms": None}
        t0 = time.perf_counter()
        ready = vlm_provider.ensure_running(env_float("VLM_TIMEOUT"))
        vlm_start_elapsed = time.perf_counter() - t0
        metrics.observe_stage("vlm_start", vlm_start_elapsed)
        yield {"stage": "vlm_start", "status": "done",
               "elapsed_ms": round(vlm_start_elapsed * 1000, 1)}
        if not ready:
            yield {"stage": "result",
                   "body": {"error": "The model service is starting up — try again shortly."},
                   "status_code": 503}
            return

    yield {"stage": "vlm", "status": "start", "elapsed_ms": None}
    start = time.perf_counter()
    answer = run_inference(image_bytes, question, history=history or None)
    inference_s = time.perf_counter() - start
    latency_ms = round(inference_s * 1000, 1)
    if not is_mock():
        metrics.observe_stage("vlm", inference_s)
        metrics.count_vlm()
        budget.record()  # count this real invocation against today's budget
    yield {"stage": "vlm", "status": "done", "elapsed_ms": latency_ms}

    # Seed the conversation on turn 1 (pins the image) and record this turn so the next
    # follow-up has the full history. `answer` is the model's reply (mock or real); in the
    # non-reveal mock path there's no meaningful answer to thread, so we still seed the
    # image + question but store the canned answer as the assistant turn for continuity.
    if not is_followup:
        conversation_store.start(conversation_id, image_bytes)
    conversation_store.append_turn(conversation_id, question, answer)

    # Rule 3: in mock mode return a disclaimer, never a fake answer — unless the
    # MOCK_REVEAL demo toggle is on, in which case show the canned answer.
    if is_mock() and not MOCK_REVEAL:
        yield {"stage": "result",
               "body": {"disclaimer": MOCK_DISCLAIMER, "mock": True, "is_chart": is_chart,
                        "chart_confidence": chart_confidence, "latency_ms": latency_ms,
                        "conversation_id": conversation_id},
               "status_code": 200}
        return

    result = {"answer": answer, "is_chart": is_chart, "chart_confidence": chart_confidence}
    if not is_mock() and not is_followup:
        # Only cache turn 1 — a follow-up's answer is history-dependent (see the cache
        # lookup above), so caching it under an (image, question) key would be unsafe.
        answer_cache.put(image_bytes, question, result)
    yield {"stage": "result",
           "body": {"mock": is_mock(), "latency_ms": latency_ms,
                    "conversation_id": conversation_id, **result},
           "status_code": 200}


@app.post("/api/ask")
def ask():
    prep, err = _prepare_ask()
    if err is not None:
        return err
    for event in _ask_events(prep):
        if event["stage"] == "result":
            return jsonify(**event["body"]), event["status_code"]
    return jsonify(error="Internal error."), 500  # pragma: no cover — _ask_events always yields a result


@app.post("/api/ask/stream")
def ask_stream():
    """Same pipeline and contract as /api/ask, streamed as Server-Sent Events so the
    frontend can show real per-stage progress (see docs/PLAN.md and _ask_events above).
    POST (not GET) because the body carries the image — browsers' native EventSource
    only supports GET, so the frontend consumes this with fetch() + a stream reader
    instead (see frontend/src/api.js askQuestionStream)."""
    prep, err = _prepare_ask()
    if err is not None:
        body, status = err
        # Mirror the same failure as a single SSE result event, so the frontend's
        # stream consumer has one code path regardless of where the pipeline stops.
        return Response(
            f"data: {json.dumps({'stage': 'result', 'body': body.get_json(), 'status_code': status})}\n\n",
            mimetype="text/event-stream",
        )

    def _sse():
        for event in _ask_events(prep):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(_sse(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # nginx: don't buffer SSE chunks
    })


if __name__ == "__main__":
    debug = env_bool("FLASK_DEBUG")
    # Pre-warm guard models off the request path so the first /api/ask is fast and the
    # Layer-3 guard is ready. With the debug reloader only the child serves
    # (WERKZEUG_RUN_MAIN=true) — warm there; without the reloader, warm unconditionally.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=warmup, daemon=True).start()
    app.run(host=env_str("HOST"), port=env_int("PORT"), debug=debug)
