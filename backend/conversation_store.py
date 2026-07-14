"""Multi-turn conversation store (Phase 5) — the server-side memory for the chatbot.

Holds, per ``conversation_id``: the chat history (a list of ``{role, text[, image_index]}``
turns) and an ORDERED LIST of the chart images the conversation is about. Turn 1 uploads
the first image; later turns may add more (ChatGPT-style) or send only the
``conversation_id`` + question, in which case the backend re-hydrates the images + prior
turns from here. Images are numbered (1-based) so the user can ask about "image 1", etc.

**Two backends behind one seam** (same shape as ``answer_cache.py``): when ``REDIS_URL``
is set and reachable, state lives in **Redis** (shared across gunicorn workers, native
TTL); otherwise an in-process dict (right for ``--dev`` / a single worker).

**Fail-open by design:** any store error is swallowed and treated as "no conversation" —
a lost conversation just means the client must re-upload / the follow-up starts fresh,
never a 500. Images live in a *separate* key from the (small) message list so a
history-only read doesn't pull the PNGs.

**TTL is refreshed on every write**, so an actively-used conversation never expires
mid-chat; only abandoned ones age out (default 30 min).

**Per-user restore (Phase 5.1):** a third namespace, ``user_key(sub) -> conversation_id``
(see ``user_key``/``get_user_conversation``/``set_user_conversation``), lets a signed-in
user's conversation survive sign-out/sign-in on a different device or after a reload —
looked up by a SHA-256 hash of the Google ``sub`` claim, never the raw id. Same TTL/
fail-open contract as everything else here.

Config (.env): ``CONVERSATION_ENABLED``, ``CONVERSATION_TTL_S`` (idle expiry),
``CONVERSATION_MAX_TURNS`` (sliding-window cap on stored user/assistant pairs).
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from threading import Lock

import redis_client
from env_config import env_bool, env_int

_ENABLED = env_bool("CONVERSATION_ENABLED")
_TTL = env_int("CONVERSATION_TTL_S")
_MAX_TURNS = env_int("CONVERSATION_MAX_TURNS")

# Hard ceiling on images STORED per conversation (bounds memory / Redis value size). The
# smaller VLM-facing cap (CONVERSATION_MAX_IMAGES) is applied by the caller at read time.
_MAX_STORED_IMAGES = 20

# Separate Redis namespaces: the (small) history JSON vs. the (large) image list, so a
# history-only read never transfers the PNGs. Both share the same conversation_id and TTL.
_RPREFIX = "conversation:"
_IPREFIX = "conversation_images:"
# user_key(sub) -> latest conversation_id, so a signed-in user's chat survives sign-out /
# sign-in (Phase 5.1). Keyed by a HASH of the Google `sub`, never the raw id (see user_key).
_UPREFIX = "conversation_user:"

# In-memory fallback. id -> (stored_at_epoch, {"messages": [...], "created_ts": float}).
_store: dict[str, tuple[float, dict]] = {}
# id -> (stored_at_epoch, [image_bytes, ...])  (ordered, oldest -> newest)
_images: dict[str, tuple[float, list[bytes]]] = {}
# user_key -> (stored_at_epoch, conversation_id)
_user_index: dict[str, tuple[float, str]] = {}
_lock = Lock()


def new_id() -> str:
    """A fresh conversation id (uuid4 hex)."""
    return uuid.uuid4().hex


def user_key(sub: str) -> str:
    """Hash a Google `sub` claim into the storage key for the user-conversation index.

    The raw `sub` never becomes a Redis key or a log line — only this hash does. Callers
    (app.py) must compute this once per request and pass the hash to
    get_user_conversation/set_user_conversation, never the raw sub.
    """
    return hashlib.sha256(sub.encode("utf-8")).hexdigest()


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
    unknown/expired/disabled. Does not include images (see :func:`get_images`)."""
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


def get_images(conversation_id: str) -> list[bytes]:
    """Return the conversation's images (oldest -> newest), or [] if none/expired."""
    if not _ENABLED or not conversation_id:
        return []
    try:
        r = redis_client.client()
        if r is not None:
            raw = r.get(_IPREFIX + conversation_id)
            if not raw:
                return []
            return [base64.b64decode(b) for b in json.loads(raw)]
        return _mem_get_images(conversation_id)
    except Exception:  # noqa: BLE001
        return []


def add_image(conversation_id: str, image_bytes: bytes) -> int:
    """Append an image to the conversation and return its 1-based index. Also seeds an
    empty message history on the first image, and refreshes the TTL. Enforces
    ``_MAX_STORED_IMAGES`` (drops the oldest past the ceiling — the returned index still
    reflects the count of images retained). Returns 0 when disabled/errored (fail-open)."""
    if not _ENABLED or not conversation_id:
        return 0
    try:
        images = get_images(conversation_id)
        images.append(image_bytes)
        if len(images) > _MAX_STORED_IMAGES:
            images = images[-_MAX_STORED_IMAGES:]
        index = len(images)  # 1-based index of the image just added (post-trim)
        r = redis_client.client()
        if r is not None:
            ex = _TTL if _TTL > 0 else None
            encoded = json.dumps([base64.b64encode(b).decode("ascii") for b in images])
            r.set(_IPREFIX + conversation_id, encoded, ex=ex)
            if not r.get(_RPREFIX + conversation_id):
                state = {"messages": [], "created_ts": time.time()}
                r.set(_RPREFIX + conversation_id, json.dumps(state), ex=ex)
            return index
        _mem_add_image(conversation_id, images)
        return index
    except Exception:  # noqa: BLE001
        return 0


def append_turn(
    conversation_id: str, question: str, answer: str, image_index: int | None = None
) -> None:
    """Append one user turn (optionally tagged with the image index it added) + one
    assistant turn, trim to the sliding window, and refresh the TTL on both the history and
    the images so an active chat never expires mid-use."""
    if not _ENABLED or not conversation_id:
        return
    try:
        state = get(conversation_id) or {"messages": [], "created_ts": time.time()}
        messages = state.get("messages", [])
        user_turn = {"role": "user", "text": question}
        if image_index:
            user_turn["image_index"] = image_index
        messages.append(user_turn)
        messages.append({"role": "assistant", "text": answer})
        state["messages"] = trim(messages)
        r = redis_client.client()
        if r is not None:
            ex = _TTL if _TTL > 0 else None
            r.set(_RPREFIX + conversation_id, json.dumps(state), ex=ex)
            if ex is not None:  # keep the images alive as long as the chat is active
                r.expire(_IPREFIX + conversation_id, ex)
            return
        _mem_append(conversation_id, state)
    except Exception:  # noqa: BLE001
        pass


def get_user_conversation(ukey: str) -> str | None:
    """Return the latest conversation_id for this user (see user_key), or None if the user
    has no live conversation (never chatted, or it expired). Fail-open like the rest of
    this module: any store error just means "nothing to restore"."""
    if not _ENABLED or not ukey:
        return None
    try:
        r = redis_client.client()
        if r is not None:
            raw = r.get(_UPREFIX + ukey)
            return raw if raw else None
        return _mem_get_user_conversation(ukey)
    except Exception:  # noqa: BLE001
        return None


def set_user_conversation(ukey: str, conversation_id: str) -> None:
    """Record/refresh which conversation is "current" for this user, so signing back in
    restores it (see app.py's _prepare_ask). Same TTL as the conversation itself — an
    abandoned chat's user-index entry expires alongside it."""
    if not _ENABLED or not ukey or not conversation_id:
        return
    try:
        r = redis_client.client()
        if r is not None:
            ex = _TTL if _TTL > 0 else None
            r.set(_UPREFIX + ukey, conversation_id, ex=ex)
            return
        _mem_set_user_conversation(ukey, conversation_id)
    except Exception:  # noqa: BLE001
        pass


def _mem_get_user_conversation(ukey: str) -> str | None:
    with _lock:
        entry = _user_index.get(ukey)
        if entry is None:
            return None
        stored_at, conversation_id = entry
        if _expired(stored_at):
            _user_index.pop(ukey, None)
            return None
        return conversation_id


def _mem_set_user_conversation(ukey: str, conversation_id: str) -> None:
    with _lock:
        _user_index[ukey] = (time.time(), conversation_id)


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


def _mem_get_images(conversation_id: str) -> list[bytes]:
    with _lock:
        entry = _images.get(conversation_id)
        if entry is None:
            return []
        stored_at, images = entry
        if _expired(stored_at):
            _images.pop(conversation_id, None)
            _store.pop(conversation_id, None)
            return []
        return list(images)


def _mem_add_image(conversation_id: str, images: list[bytes]) -> None:
    with _lock:
        now = time.time()
        _images[conversation_id] = (now, list(images))
        if conversation_id not in _store:
            _store[conversation_id] = (now, {"messages": [], "created_ts": now})


def _mem_append(conversation_id: str, state: dict) -> None:
    with _lock:
        now = time.time()
        _store[conversation_id] = (now, state)
        imgs = _images.get(conversation_id)
        if imgs is not None:
            _images[conversation_id] = (now, imgs[1])  # refresh image TTL too


def reset() -> None:
    """Drop all in-memory state (used by tests)."""
    with _lock:
        _store.clear()
        _images.clear()
        _user_index.clear()
