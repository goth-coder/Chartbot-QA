"""Multi-turn conversation store (Phase 5) — the server-side memory for the chatbot.

Holds, per ``conversation_id``: the chat history (a list of ``{role, text}`` turns) and
the sanitized chart image the whole conversation is about. On turn 1 the client uploads
the image; on follow-up turns it sends only ``conversation_id`` + question, and the
backend re-hydrates the image + prior turns from here.

**Two backends behind one seam** (same shape as ``answer_cache.py``): when ``REDIS_URL``
is set and reachable, state lives in **Redis** (shared across gunicorn workers, native
TTL); otherwise an in-process dict (right for ``--dev`` / a single worker).

**Fail-open by design:** any store error is swallowed and treated as "no conversation" —
a lost conversation just means the client must re-upload the image / the follow-up starts
fresh, never a 500. The image is base64-encoded in a *separate* key from the (small)
message list so a re-hydrate that only needs history doesn't pull the whole PNG.

**TTL is refreshed on every write** (``touch``), so an actively-used conversation never
expires mid-chat; only abandoned ones age out (default 30 min).

Config (.env): ``CONVERSATION_ENABLED``, ``CONVERSATION_TTL_S`` (idle expiry),
``CONVERSATION_MAX_TURNS`` (sliding-window cap on stored user/assistant pairs).
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from threading import Lock

import redis_client
from env_config import env_bool, env_int

_ENABLED = env_bool("CONVERSATION_ENABLED")
_TTL = env_int("CONVERSATION_TTL_S")
_MAX_TURNS = env_int("CONVERSATION_MAX_TURNS")

# Separate Redis namespaces: the (small) history JSON vs. the (large) image bytes, so a
# history-only read never transfers the PNG. Both share the same conversation_id and TTL.
_RPREFIX = "conversation:"
_IPREFIX = "conversation_image:"

# In-memory fallback. id -> (stored_at_epoch, {"messages": [...], "created_ts": float}).
_store: dict[str, tuple[float, dict]] = {}
# id -> (stored_at_epoch, image_bytes)
_images: dict[str, tuple[float, bytes]] = {}
_lock = Lock()


def new_id() -> str:
    """A fresh conversation id (uuid4 hex)."""
    return uuid.uuid4().hex


def _expired(stored_at: float) -> bool:
    return _TTL > 0 and (time.time() - stored_at) > _TTL


def trim(messages: list[dict]) -> list[dict]:
    """Sliding window: keep only the last ``_MAX_TURNS`` user/assistant *pairs* (i.e. the
    last ``2 * _MAX_TURNS`` messages). Charts rarely need more than a few recent turns, and
    this caps the token budget fed back to the VLM."""
    if _MAX_TURNS <= 0:
        return messages
    cap = _MAX_TURNS * 2
    return messages[-cap:] if len(messages) > cap else messages


def get(conversation_id: str) -> dict | None:
    """Return ``{"messages": [...], "created_ts": float}`` for this id, or None if
    unknown/expired/disabled. Does not include the image (see :func:`get_image`)."""
    if not _ENABLED or not conversation_id:
        return None
    try:
        r = redis_client.client()
        if r is not None:
            raw = r.get(_RPREFIX + conversation_id)
            return json.loads(raw) if raw else None
        return _mem_get(conversation_id)
    except Exception:  # noqa: BLE001 — the store must never break a request
        return None


def get_image(conversation_id: str) -> bytes | None:
    """Return the sanitized image bytes for this conversation, or None."""
    if not _ENABLED or not conversation_id:
        return None
    try:
        r = redis_client.client()
        if r is not None:
            raw = r.get(_IPREFIX + conversation_id)
            return base64.b64decode(raw) if raw else None
        return _mem_get_image(conversation_id)
    except Exception:  # noqa: BLE001
        return None


def start(conversation_id: str, image_bytes: bytes) -> None:
    """Begin a conversation: store its image and an empty history. Idempotent — re-starting
    an existing id just refreshes the image + TTL (harmless if the client retries turn 1)."""
    if not _ENABLED or not conversation_id:
        return
    try:
        r = redis_client.client()
        if r is not None:
            ex = _TTL if _TTL > 0 else None
            r.set(_IPREFIX + conversation_id, base64.b64encode(image_bytes).decode("ascii"), ex=ex)
            existing = r.get(_RPREFIX + conversation_id)
            if not existing:
                state = {"messages": [], "created_ts": time.time()}
                r.set(_RPREFIX + conversation_id, json.dumps(state), ex=ex)
            return
        _mem_start(conversation_id, image_bytes)
    except Exception:  # noqa: BLE001
        pass


def append_turn(conversation_id: str, question: str, answer: str) -> None:
    """Append one user turn + one assistant turn, trim to the sliding window, and refresh
    the TTL on both the history and the image so an active chat never expires mid-use."""
    if not _ENABLED or not conversation_id:
        return
    try:
        state = get(conversation_id) or {"messages": [], "created_ts": time.time()}
        messages = state.get("messages", [])
        messages.append({"role": "user", "text": question})
        messages.append({"role": "assistant", "text": answer})
        state["messages"] = trim(messages)
        r = redis_client.client()
        if r is not None:
            ex = _TTL if _TTL > 0 else None
            r.set(_RPREFIX + conversation_id, json.dumps(state), ex=ex)
            # Refresh the image key's TTL too (it's the same conversation staying alive).
            if ex is not None:
                r.expire(_IPREFIX + conversation_id, ex)
            return
        _mem_append(conversation_id, state)
    except Exception:  # noqa: BLE001
        pass


def _mem_get(conversation_id: str) -> dict | None:
    with _lock:
        entry = _store.get(conversation_id)
        if entry is None:
            return None
        stored_at, state = entry
        if _expired(stored_at):
            _store.pop(conversation_id, None)
            _images.pop(conversation_id, None)
            return None
        return json.loads(json.dumps(state))  # deep copy so callers can't mutate the store


def _mem_get_image(conversation_id: str) -> bytes | None:
    with _lock:
        entry = _images.get(conversation_id)
        if entry is None:
            return None
        stored_at, image_bytes = entry
        if _expired(stored_at):
            _images.pop(conversation_id, None)
            _store.pop(conversation_id, None)
            return None
        return image_bytes


def _mem_start(conversation_id: str, image_bytes: bytes) -> None:
    with _lock:
        now = time.time()
        _images[conversation_id] = (now, image_bytes)
        if conversation_id not in _store:
            _store[conversation_id] = (now, {"messages": [], "created_ts": now})


def _mem_append(conversation_id: str, state: dict) -> None:
    with _lock:
        now = time.time()
        _store[conversation_id] = (now, state)
        img = _images.get(conversation_id)
        if img is not None:
            _images[conversation_id] = (now, img[1])  # refresh image TTL too


def reset() -> None:
    """Drop all in-memory state (used by tests)."""
    with _lock:
        _store.clear()
        _images.clear()
