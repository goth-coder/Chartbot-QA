"""Unit tests for the chart gate (chart_check.py).

These run WITHOUT torch / transformers and WITHOUT any model download: the CLIP
stage is monkeypatched, so the tests are fast and pass on a machine that only
has the light deps. They exercise the decision logic in ``looks_like_chart``:
threshold comparison and the fallback to the pixel heuristic.
"""

import io

from PIL import Image

import chart_check


def _png_bytes(color="white", size=(32, 32)) -> bytes:
    """A tiny in-memory PNG for the heuristic path."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()




def test_high_prob_is_chart(monkeypatch):
    # CLIP says 0.9 -> above threshold -> (True, 0.9)
    monkeypatch.setattr(chart_check, "_clip_chart_prob", lambda b: 0.9)
    assert chart_check.looks_like_chart(b"ignored") == (True, 0.9)


def test_low_prob_not_chart(monkeypatch):
    # CLIP says 0.1 -> below threshold -> (False, 0.1)
    monkeypatch.setattr(chart_check, "_clip_chart_prob", lambda b: 0.1)
    assert chart_check.looks_like_chart(b"ignored") == (False, 0.1)


def test_threshold_is_inclusive(monkeypatch):
    # Exactly at the threshold counts as a chart.
    monkeypatch.setattr(
        chart_check, "_clip_chart_prob", lambda b: chart_check._CLIP_THRESHOLD
    )
    is_chart, _ = chart_check.looks_like_chart(b"ignored")
    assert is_chart is True


def test_none_prob_falls_back_to_heuristic(monkeypatch):
    # CLIP unavailable (returns None) -> use the heuristic's verdict verbatim.
    monkeypatch.setattr(chart_check, "_clip_chart_prob", lambda b: None)
    sentinel = (True, 0.42)
    monkeypatch.setattr(chart_check, "_heuristic_chart", lambda b: sentinel)
    assert chart_check.looks_like_chart(b"ignored") is sentinel


def test_clip_confident_is_chart_without_ocr(monkeypatch):
    # CLIP is confident it's a chart -> chart, regardless of OCR. The OCR "has numeric
    # data" veto was removed (2026-07-13): it falsely rejected real charts whose values
    # are labels (e.g. a horizontal bar chart of country names) or whose axis digits
    # Tesseract can't read. So even with no OCR-legible digits, a CLIP-confident chart
    # is a chart.
    monkeypatch.setattr(chart_check, "_clip_chart_prob", lambda b: 0.95)
    monkeypatch.setattr(chart_check, "_has_data_values", lambda b: False)
    is_chart, conf = chart_check.looks_like_chart(b"img")
    assert is_chart is True
    assert conf == 0.95


def test_clip_below_threshold_is_not_chart(monkeypatch):
    # CLIP unsure it's a chart -> not a chart (the confidence, P(chart), is returned).
    monkeypatch.setattr(chart_check, "_clip_chart_prob", lambda b: 0.2)
    is_chart, conf = chart_check.looks_like_chart(b"img")
    assert is_chart is False
    assert conf == 0.2


def test_fallback_when_clip_cannot_load(monkeypatch):
    # Force the model to be unavailable: _load_clip returns None, so the real
    # _clip_chart_prob returns None and looks_like_chart runs the actual pixel
    # heuristic — and must still return a (bool, float) with no ML deps.
    monkeypatch.setattr(chart_check, "_load_clip", lambda: None)
    is_chart, confidence = chart_check.looks_like_chart(_png_bytes())
    assert isinstance(is_chart, bool)
    assert isinstance(confidence, float)
