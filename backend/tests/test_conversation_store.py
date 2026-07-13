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


def test_add_image_returns_index_and_seeds_history():
    cid = cs.new_id()
    assert cs.add_image(cid, b"png-1") == 1        # 1-based index
    assert cs.get_images(cid) == [b"png-1"]
    state = cs.get(cid)
    assert state is not None
    assert state["messages"] == []


def test_multiple_images_accumulate_in_order():
    cid = cs.new_id()
    assert cs.add_image(cid, b"png-1") == 1
    assert cs.add_image(cid, b"png-2") == 2
    assert cs.add_image(cid, b"png-3") == 3
    assert cs.get_images(cid) == [b"png-1", b"png-2", b"png-3"]


def test_stored_images_capped_drops_oldest(monkeypatch):
    monkeypatch.setattr(cs, "_MAX_STORED_IMAGES", 2)
    cid = cs.new_id()
    cs.add_image(cid, b"a")
    cs.add_image(cid, b"b")
    cs.add_image(cid, b"c")  # exceeds the ceiling -> oldest ("a") drops
    assert cs.get_images(cid) == [b"b", b"c"]


def test_unknown_id_returns_none_and_empty():
    assert cs.get("nope") is None
    assert cs.get_images("nope") == []


def test_append_turn_records_image_index():
    cid = cs.new_id()
    cs.add_image(cid, b"img1")
    cs.append_turn(cid, "what is the max?", "9", image_index=1)
    cs.append_turn(cid, "and the min?", "1")  # no new image this turn
    msgs = cs.get(cid)["messages"]
    assert msgs == [
        {"role": "user", "text": "what is the max?", "image_index": 1},
        {"role": "assistant", "text": "9"},
        {"role": "user", "text": "and the min?"},
        {"role": "assistant", "text": "1"},
    ]


def test_sliding_window_keeps_last_n_pairs():
    cid = cs.new_id()
    cs.add_image(cid, b"img")
    for i in range(5):  # _MAX_TURNS == 3 pairs -> keep last 6 messages
        cs.append_turn(cid, f"q{i}", f"a{i}")
    msgs = cs.get(cid)["messages"]
    assert len(msgs) == 6                        # 3 pairs
    assert msgs[0] == {"role": "user", "text": "q2"}   # q0, q1 dropped
    assert msgs[-1] == {"role": "assistant", "text": "a4"}


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(cs, "_ENABLED", False)
    cid = cs.new_id()
    assert cs.add_image(cid, b"img") == 0        # no-op when disabled
    assert cs.get(cid) is None
    assert cs.get_images(cid) == []


def test_get_returns_a_copy():
    cid = cs.new_id()
    cs.add_image(cid, b"img")
    cs.append_turn(cid, "q", "a")
    state = cs.get(cid)
    state["messages"].append({"role": "user", "text": "injected"})
    assert len(cs.get(cid)["messages"]) == 2      # store not mutated by the caller


def test_ttl_expiry():
    cid = cs.new_id()
    cs.add_image(cid, b"img")
    assert cs.get(cid) is not None
    # Force the stored timestamp into the past instead of sleeping.
    with cs._lock:
        cs._TTL = 1  # module const; _expired uses > _TTL seconds
        state = cs._store[cid][1]
        cs._store[cid] = (time.time() - 5, state)
        imgs = cs._images[cid][1]
        cs._images[cid] = (time.time() - 5, imgs)
    try:
        assert cs.get(cid) is None
        assert cs.get_images(cid) == []
    finally:
        cs._TTL = 0
