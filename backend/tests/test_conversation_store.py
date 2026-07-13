"""Tests for the multi-turn conversation store (real in-memory path; Redis unavailable
in tests since .env.example leaves REDIS_URL empty, so redis_client.client() returns None
and the module falls back to its in-process dict — the real fallback code, not a fake)."""
import time

import pytest

import conversation_store as cs


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Clean store + deterministic small config for each test."""
    cs.reset()
    monkeypatch.setattr(cs, "_ENABLED", True)
    monkeypatch.setattr(cs, "_TTL", 0)          # never expire by default (override per-test)
    monkeypatch.setattr(cs, "_MAX_TURNS", 3)
    yield
    cs.reset()


def test_new_id_is_unique():
    assert cs.new_id() != cs.new_id()


def test_start_then_get_image_and_empty_history():
    cid = cs.new_id()
    cs.start(cid, b"png-bytes")
    assert cs.get_image(cid) == b"png-bytes"
    state = cs.get(cid)
    assert state is not None
    assert state["messages"] == []


def test_unknown_id_returns_none():
    assert cs.get("nope") is None
    assert cs.get_image("nope") is None


def test_append_turn_accumulates_user_and_assistant():
    cid = cs.new_id()
    cs.start(cid, b"img")
    cs.append_turn(cid, "what is the max?", "9")
    cs.append_turn(cid, "and the min?", "1")
    msgs = cs.get(cid)["messages"]
    assert msgs == [
        {"role": "user", "text": "what is the max?"},
        {"role": "assistant", "text": "9"},
        {"role": "user", "text": "and the min?"},
        {"role": "assistant", "text": "1"},
    ]


def test_sliding_window_keeps_last_n_pairs():
    cid = cs.new_id()
    cs.start(cid, b"img")
    for i in range(5):  # _MAX_TURNS == 3 pairs -> keep last 6 messages
        cs.append_turn(cid, f"q{i}", f"a{i}")
    msgs = cs.get(cid)["messages"]
    assert len(msgs) == 6                        # 3 pairs
    assert msgs[0] == {"role": "user", "text": "q2"}   # q0, q1 dropped
    assert msgs[-1] == {"role": "assistant", "text": "a4"}


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(cs, "_ENABLED", False)
    cid = cs.new_id()
    cs.start(cid, b"img")
    assert cs.get(cid) is None
    assert cs.get_image(cid) is None


def test_get_returns_a_copy():
    cid = cs.new_id()
    cs.start(cid, b"img")
    cs.append_turn(cid, "q", "a")
    state = cs.get(cid)
    state["messages"].append({"role": "user", "text": "injected"})
    assert len(cs.get(cid)["messages"]) == 2      # store not mutated by the caller


def test_ttl_expiry(monkeypatch):
    monkeypatch.setattr(cs, "_TTL", 1)  # _expired uses > _TTL seconds; use a tiny sleep buffer
    cid = cs.new_id()
    cs.start(cid, b"img")
    assert cs.get(cid) is not None
    # Force the stored timestamp into the past instead of sleeping a full second.
    with cs._lock:
        state = cs._store[cid][1]
        cs._store[cid] = (time.time() - 5, state)
        img = cs._images[cid][1]
        cs._images[cid] = (time.time() - 5, img)
    assert cs.get(cid) is None
    assert cs.get_image(cid) is None
