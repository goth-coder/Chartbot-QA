"""Standalone GPU VLM inference service — Qwen3-VL-8B (+ optional LoRA) over HTTP.

The production counterpart to the backend's remote serving mode (see
``backend/model_adapter._predict_remote``): the CPU gatekeeper (Layer-1/2/3 guards +
chart gate) runs the cheap checks first, then POSTs the surviving ``{image, question}``
here. This runs on a GPU box as a **scale-to-zero (min=0)** service behind the
gatekeeper — Qwen-8B does not fit the 6 GB 4050 (roadmap §B2), so this is where it lives
(a cloud GPU, or a big local GPU).

Contract (matches ``_predict_remote``):
    POST /predict  {"images": ["<base64>", ...], "question": "...",
                    "response_mode": "reasoned|direct"?, "max_new_tokens": <int>?,
                    "history": [...]?}  -> {"answer": "..."}
    GET  /health   -> {"status": "ok", "model": "<id>", "adapter": "<path|null>"}

Response BEHAVIOR (which prompt / how many tokens) is per-request, not per-deploy: the
backend picks a NAMED ``response_mode`` + token budget and sends them; this service
validates the mode against ``response_modes`` (allowlist + clamp) and applies the model's
chat template. Do NOT re-introduce a baked-in ``QWEN_ANSWER_SUFFIX`` / ``QWEN_MAX_NEW_TOKENS``
here — that drift (backend config the remote path ignored) is exactly what this replaced.

The Qwen wrapper + response-mode registry are imported from the installable modeling
package (``pip install -e ./modeling``) — deliberately NOT re-vendored here — so the
loading/generation logic has a single source of truth (CLAUDE.md: no new vendored copies).

Config via env (no in-code defaults for the model-LOAD knobs — those ARE infra config;
behavior knobs are per-request, above):
    QWEN_MODEL_ID, QWEN_ADAPTER_PATH ('' = base model), QWEN_QUANTIZATION (none|8bit|4bit)
    VLM_HOST (default 0.0.0.0), VLM_PORT (default 8001)   # operational, defaults allowed
"""
from __future__ import annotations

import base64
import io
import logging
import os

from flask import Flask, jsonify, request
from PIL import Image

from chartqa import response_modes
from chartqa.models.qwen_vl_chat import QwenVLChat

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vlm_service")


def _require(name: str) -> str:
    """Required env var — fail fast at boot with a clear message (no in-code defaults)."""
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(f"Missing required env var {name!r} (see vlm_service/README.md).")
    return val


# --- Warm the model ONCE at import (boot), never in the request path. ---
# Only model-LOAD knobs live in env here; response behavior (prompt/tokens) is per-request.
_MODEL_ID = _require("QWEN_MODEL_ID")
_ADAPTER = _require("QWEN_ADAPTER_PATH").strip() or None
_QUANT = _require("QWEN_QUANTIZATION")

log.info("Loading %s (adapter=%s, quantization=%s) ...", _MODEL_ID, _ADAPTER, _QUANT)
_CHAT = QwenVLChat(model_name=_MODEL_ID, adapter_path=_ADAPTER, quantization=_QUANT)
log.info("Model ready — serving.")

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", model=_MODEL_ID, adapter=_ADAPTER)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    # Accept `images` (list of base64, oldest -> newest) for multi-chart conversations;
    # fall back to a single `image` for the classic one-chart contract.
    images_b64 = data.get("images")
    if not images_b64:
        single = data.get("image")
        images_b64 = [single] if single else []
    question = (data.get("question") or "").strip()
    # Optional multi-turn history: prior [{"role","text"}] turns. build_messages numbers
    # the images and attaches them once, in a leading turn.
    history = data.get("history") or None
    if not images_b64 or not question:
        return jsonify(error="'images' (base64 list) and 'question' are required."), 400

    # Behavior is per-request: validate the requested mode against the allowlist (unknown
    # -> default, never errors) and clamp the token budget — never trust a raw request
    # value. The mode becomes a system prompt, NOT a suffix on the user's question text.
    system_prompt, max_new_tokens = response_modes.resolve(
        data.get("response_mode"), data.get("max_new_tokens")
    )

    pil_images = [
        Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in images_b64
    ]
    raw = _CHAT.chat(
        images=pil_images, text=question, system_prompt=system_prompt,
        max_new_tokens=max_new_tokens, history=history,
    )
    # Same post-processing as the in-process path so both modes return identical answers:
    # keep only the terse final answer after the last "Answer:" the reasoning may emit.
    answer = raw.split("Answer:")[-1].strip()
    return jsonify(answer=answer)


if __name__ == "__main__":
    # Dev entrypoint. In prod use gunicorn (see README) — the model warms at import, so a
    # single worker keeps one copy in VRAM: `gunicorn -w 1 -t 300 -b :8001 server:app`.
    # Cloud Run (and Cloud Run GPU) inject $PORT and require the container to listen on
    # exactly that port — checked first so the same image runs unmodified there; RunPod/
    # local dev has no $PORT, so VLM_PORT (or 8001) applies instead.
    port = int(os.environ.get("PORT", os.environ.get("VLM_PORT", "8001")))
    app.run(host=os.environ.get("VLM_HOST", "0.0.0.0"), port=port)
