"""Unit tests for the cheap on-topic classifier (topic_check.py — embedding similarity).

Monkeypatch _load_model/_embed/_reference_embeddings so this runs with no ML deps
installed, matching test_guard.py's convention for the other detectors. Real-model
accuracy (precision/recall against real ChartQA questions + varied negatives) was
validated experimentally before choosing the threshold — see topic_check.py's module
docstring for that calibration; these tests only prove the wiring/fail-open logic.
"""

import torch

import topic_check as tc


def _unit_vector(score: float):
    """A 2-D unit vector whose dot product with (1, 0) equals `score` exactly."""
    score = max(-1.0, min(1.0, score))
    return [score, (1 - score * score) ** 0.5]


def _stub(monkeypatch, question: str, score: float):
    """Wire _embed/_reference_embeddings so on_topic_confidence(question) == score,
    without touching any real model."""
    monkeypatch.setattr(tc, "TOPIC_CHECK_ENABLED", True)
    monkeypatch.setattr(tc, "_load_model", lambda: object())
    monkeypatch.setattr(tc, "_reference_embeddings", lambda: torch.tensor([[1.0, 0.0]]))

    def _embed(texts):
        assert texts == [question]
        return torch.tensor([_unit_vector(score)])

    monkeypatch.setattr(tc, "_embed", _embed)


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_ENABLED", False)
    assert tc.on_topic_confidence("what was the peak value?") is None
    assert tc.is_confidently_on_topic("what was the peak value?") is False


def test_unavailable_model_returns_none(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_ENABLED", True)
    monkeypatch.setattr(tc, "_load_model", lambda: None)
    assert tc.on_topic_confidence("what was the peak value?") is None
    assert tc.is_confidently_on_topic("what was the peak value?") is False


def test_on_topic_question_scores_high(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_THRESHOLD", 0.40)
    q = "What was the peak value in the chart?"
    _stub(monkeypatch, q, 0.9)
    assert tc.on_topic_confidence(q) == pytest_approx(0.9)
    assert tc.is_confidently_on_topic(q) is True


def test_off_topic_question_scores_low(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_THRESHOLD", 0.40)
    q = "Who won the 2024 election?"
    _stub(monkeypatch, q, 0.1)
    assert tc.on_topic_confidence(q) == pytest_approx(0.1)
    assert tc.is_confidently_on_topic(q) is False


def test_ambiguous_score_is_not_confidently_on_topic(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_THRESHOLD", 0.40)
    q = "tell me something interesting"
    _stub(monkeypatch, q, 0.35)  # below threshold, but not near-zero either
    assert tc.is_confidently_on_topic(q) is False


def test_score_exactly_at_threshold_counts_as_confident(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_THRESHOLD", 0.40)
    q = "borderline question"
    _stub(monkeypatch, q, 0.40)
    assert tc.is_confidently_on_topic(q) is True


def test_embedding_error_fails_open_to_none(monkeypatch):
    monkeypatch.setattr(tc, "TOPIC_CHECK_ENABLED", True)
    monkeypatch.setattr(tc, "_load_model", lambda: object())

    def _boom(texts):
        raise RuntimeError("model error")

    monkeypatch.setattr(tc, "_embed", _boom)
    assert tc.on_topic_confidence("what was the peak value?") is None
    assert tc.is_confidently_on_topic("what was the peak value?") is False


def test_warmup_noop_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(tc, "_load_model", lambda: None)
    tc.warmup()  # must not raise


def test_is_available_reflects_load_state(monkeypatch):
    monkeypatch.setattr(tc, "_load_model", lambda: None)
    assert tc.is_available() is False
    monkeypatch.setattr(tc, "_load_model", lambda: object())
    assert tc.is_available() is True


def pytest_approx(x, tol=1e-5):
    class _Approx(float):
        def __eq__(self, other):
            return abs(float(other) - float(self)) < tol

    return _Approx(x)
