"""Tests for the response-mode resolver (backend/response_modes.py — the vendored copy;
byte-identical to modeling/chartqa/response_modes.py per test_vendor_sync)."""

import response_modes as rm


def test_reasoned_mode_has_system_prompt_and_default_budget():
    system_prompt, tokens = rm.resolve("reasoned", None)
    assert system_prompt is not None and "Answer:" in system_prompt
    assert tokens == rm._MODES["reasoned"]["default_max_new_tokens"]


def test_direct_mode_has_no_system_prompt():
    system_prompt, tokens = rm.resolve("direct", None)
    assert system_prompt is None
    assert tokens == rm._MODES["direct"]["default_max_new_tokens"]


def test_unknown_mode_falls_back_to_default_never_raises():
    system_prompt, tokens = rm.resolve("totally-made-up", None)
    default_prompt, default_tokens = rm.resolve(rm.DEFAULT_MODE, None)
    assert system_prompt == default_prompt
    assert tokens == default_tokens


def test_none_mode_uses_default():
    assert rm.resolve(None, None) == rm.resolve(rm.DEFAULT_MODE, None)


def test_explicit_token_override_is_honored():
    _, tokens = rm.resolve("reasoned", 100)
    assert tokens == 100


def test_token_budget_clamped_to_ceiling():
    _, tokens = rm.resolve("reasoned", 99999)
    assert tokens == rm._MAX_MAX_NEW_TOKENS


def test_token_budget_clamped_to_floor():
    _, tokens = rm.resolve("reasoned", 1)
    assert tokens == rm._MIN_MAX_NEW_TOKENS


def test_default_mode_is_reasoned():
    # The whole point of the change: chart arithmetic needs reasoning by default.
    assert rm.DEFAULT_MODE == "reasoned"
