"""Tests for the feedback -> GCS flywheel.

The GCS client is faked at its real boundary (feedback_store._bucket), never the
record() function itself — record()'s own logic (image dedupe, JSONL append, fail-open)
is what we exercise. Writes run in a daemon thread, so tests join it before asserting."""
import json

import pytest

import feedback_store as fs


class _FakeBlob:
    def __init__(self, store, path):
        self._store = store
        self._path = path

    def exists(self):
        return self._path in self._store

    def upload_from_string(self, data, content_type=None):
        self._store[self._path] = data

    def download_as_text(self):
        return self._store[self._path]


class _FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, path):
        return _FakeBlob(self.objects, path)


@pytest.fixture()
def fake_bucket(monkeypatch):
    bucket = _FakeBucket()
    monkeypatch.setattr(fs, "_ENABLED", True)
    monkeypatch.setattr(fs, "_BUCKET", "test-bucket")
    monkeypatch.setattr(fs, "_bucket", lambda: bucket)
    return bucket


def _record_and_join(**kwargs):
    """Run record() and block on the spawned daemon thread so assertions see the writes."""
    import threading

    before = set(threading.enumerate())
    fs.record(**kwargs)
    for t in threading.enumerate():
        if t not in before:
            t.join(timeout=5)


def test_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(fs, "_ENABLED", False)
    # No bucket touched, no exception — record must be a silent no-op.
    fs.record(
        conversation_id="c1", question="q", model_answer="a", vote="up", note="",
        image_bytes=b"img",
    )


def test_no_bucket_configured_is_noop(monkeypatch):
    monkeypatch.setattr(fs, "_ENABLED", True)
    monkeypatch.setattr(fs, "_BUCKET", "")
    fs.record(
        conversation_id="c1", question="q", model_answer="a", vote="down", note="wrong",
        image_bytes=b"img",
    )


def test_record_writes_image_and_vote_object(fake_bucket):
    _record_and_join(
        conversation_id="c1", question="max?", model_answer="9", vote="up", note="",
        image_bytes=b"png-bytes",
    )
    # An image object and one per-vote json object were written.
    paths = list(fake_bucket.objects)
    assert any(p.startswith("feedback/images/") and p.endswith(".png") for p in paths)
    vote_path = next(
        p for p in paths if p.startswith("feedback/") and p.endswith(".json")
        and not p.startswith("feedback/images/")
    )
    row = json.loads(fake_bucket.objects[vote_path])
    assert row["conversation_id"] == "c1"
    assert row["question"] == "max?"
    assert row["model_answer"] == "9"
    assert row["vote"] == "up"
    assert row["image_path"].startswith("feedback/images/")


def test_image_written_once_but_one_object_per_vote(fake_bucket):
    # Two votes on the same image: the image object is written once, each vote gets its
    # own object (no shared file to race on).
    for note in ("first", "second"):
        _record_and_join(
            conversation_id="c1", question="q", model_answer="a", vote="down", note=note,
            image_bytes=b"same-image",
        )
    image_objs = [p for p in fake_bucket.objects if p.startswith("feedback/images/")]
    assert len(image_objs) == 1
    vote_objs = [
        p for p in fake_bucket.objects
        if p.endswith(".json") and not p.startswith("feedback/images/")
    ]
    assert len(vote_objs) == 2


def test_fail_open_when_bucket_raises(monkeypatch):
    monkeypatch.setattr(fs, "_ENABLED", True)
    monkeypatch.setattr(fs, "_BUCKET", "test-bucket")

    class _Boom:
        def blob(self, path):
            raise RuntimeError("gcs down")

    monkeypatch.setattr(fs, "_bucket", lambda: _Boom())
    # Must not raise — a GCS failure is logged and swallowed.
    _record_and_join(
        conversation_id="c1", question="q", model_answer="a", vote="up", note="",
        image_bytes=b"img",
    )
