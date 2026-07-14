"""Feedback flywheel (Phase 5) — persist 👍/👎 votes + the chart image to GCS as a
durable, append-only fine-tuning data source.

Each vote appends one JSON line to ``gs://<bucket>/feedback/YYYY-MM-DD.jsonl`` and writes
the sanitized chart PNG once to ``gs://<bucket>/feedback/images/<sha256>.png`` (skipped if
that object already exists). Later, batch-download the JSONL + images, curate the 👎 rows
(write the correct answer), and assemble a ``feedback_vN.jsonl`` training set — gated by
the same eval bar as any other run. **No per-vote auto-training.**

**Fire-and-forget, fail-open by design:** :func:`record` returns immediately after
spawning a daemon thread; any error (no bucket configured, package missing, GCS write
failure) is logged and swallowed. Feedback is nice-to-have telemetry — it must never break
the user's request path.

Config (.env): ``FEEDBACK_ENABLED`` (default 0), ``FEEDBACK_GCS_BUCKET`` (empty default).
The backend service account needs object write access on the bucket
(``roles/storage.objectAdmin``, or objectCreator+objectViewer).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid

from env_config import env_bool, env_str

log = logging.getLogger("feedback_store")

_ENABLED = env_bool("FEEDBACK_ENABLED")
_BUCKET = env_str("FEEDBACK_GCS_BUCKET").strip()

# Lazily-created, memoized GCS client — created once on first successful use, never in the
# request path (the record() call returns before the client is even touched).
_client = None
_client_lock = threading.Lock()


def _bucket():
    """Return a google-cloud-storage Bucket handle, or None if unavailable/misconfigured.

    Memoized. Any import/credential error returns None so the caller fails open."""
    global _client
    if not _BUCKET:
        return None
    with _client_lock:
        if _client is None:
            try:
                from google.cloud import storage  # local import: optional dependency
                _client = storage.Client()
            except Exception as exc:  # noqa: BLE001
                log.warning("Feedback GCS client unavailable (%s); feedback not persisted.", exc)
                return None
        try:
            return _client.bucket(_BUCKET)
        except Exception as exc:  # noqa: BLE001
            log.warning("Feedback GCS bucket %r unavailable (%s).", _BUCKET, exc)
            return None


def record(
    conversation_id: str,
    question: str,
    model_answer: str,
    vote: str,
    note: str,
    image_bytes: bytes,
) -> None:
    """Persist one feedback record. Returns immediately; the GCS write runs in a daemon
    thread. No-op when disabled / no bucket configured."""
    if not _ENABLED or not _BUCKET:
        return
    # Copy the args into the thread's closure; image_bytes is already an immutable bytes.
    threading.Thread(
        target=_record_bg,
        args=(conversation_id, question, model_answer, vote, note, image_bytes),
        daemon=True,
    ).start()


def _record_bg(
    conversation_id: str,
    question: str,
    model_answer: str,
    vote: str,
    note: str,
    image_bytes: bytes,
) -> None:
    try:
        bucket = _bucket()
        if bucket is None:
            return
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        image_path = f"feedback/images/{image_sha256}.png"

        # Write the image once — many votes can share the same chart, so skip if it exists.
        img_blob = bucket.blob(image_path)
        if not img_blob.exists():
            img_blob.upload_from_string(image_bytes, content_type="image/png")

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record = {
            "ts": ts,
            "conversation_id": conversation_id,
            "question": question,
            "model_answer": model_answer,
            "vote": vote,
            "note": note,
            "image_path": image_path,
            "image_sha256": image_sha256,
        }
        # One immutable object per vote under a daily prefix — NOT an append to a shared
        # daily file. GCS objects can't be appended to, so a read-modify-rewrite would race
        # (two concurrent votes both read, both rewrite, the second clobbers the first).
        # Per-vote objects are trivially concurrent-safe; compact them into a single
        # training .jsonl offline when building a fine-tune set (see the roadmap MLOps loop).
        day = time.strftime("%Y-%m-%d", time.gmtime())
        vote_path = f"feedback/{day}/{uuid.uuid4().hex}.json"
        bucket.blob(vote_path).upload_from_string(
            json.dumps(record, ensure_ascii=False),
            content_type="application/json",
        )
        log.info("Feedback recorded (%s) for conversation %s -> %s", vote, conversation_id, vote_path)
    except Exception as exc:  # noqa: BLE001 — feedback must never break anything
        log.warning("Feedback record failed (%s); dropped.", exc)


def reset() -> None:
    """Drop the memoized client (used by tests)."""
    global _client
    with _client_lock:
        _client = None
