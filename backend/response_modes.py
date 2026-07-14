# --- vendor-sync:ignore-start ---
"""Named VLM response modes — vendored copy for the backend serving image.

Source of truth: modeling/chartqa/response_modes.py. This copy is kept self-contained so
the backend Docker image builds without the modeling tree. The two files are kept in
sync by backend/tests/test_vendor_sync.py, which fails if the shared logic (everything
outside the vendor-sync:ignore markers) diverges — edit both, or the test fails.
"""
# --- vendor-sync:ignore-end ---
from __future__ import annotations

# The "reasoned" (chain-of-thought) system prompt. Charts encode values the VLM reads as
# approximate visual patterns; asking for a value + a calculation "directly" compounds
# that imprecision into wrong arithmetic. Telling the model to state the value(s) first
# and compute step by step, then end with a single "Answer:" line, measurably improves
# comparison/difference/ratio questions. The caller keeps only the text after the LAST
# "Answer:" (raw.split("Answer:")[-1]), so the user still sees a terse final answer.
_REASONED_SYSTEM_PROMPT = (
    "You answer questions about charts. If the question needs a calculation "
    "(a difference, sum, ratio, or comparison), first read and state the relevant "
    "value(s) from the chart, then compute step by step. Then, on a new line, write "
    "exactly 'Answer:' followed by only the short final answer (1-10 words). If no "
    "calculation is needed, you may write the 'Answer:' line directly."
)

# Named modes. Each maps to a system prompt (None = no reasoning scaffold) and a default
# generation budget. "reasoned" needs headroom for the reasoning text BEFORE the final
# "Answer:" line; "direct" is terse and cheap.
_MODES = {
    "direct": {"system_prompt": None, "default_max_new_tokens": 64},
    "reasoned": {"system_prompt": _REASONED_SYSTEM_PROMPT, "default_max_new_tokens": 256},
}

DEFAULT_MODE = "reasoned"

# Clamp bounds for a caller-supplied token budget — never trust a raw request value.
_MIN_MAX_NEW_TOKENS = 16
_MAX_MAX_NEW_TOKENS = 512


def resolve(mode: str | None, max_new_tokens: int | None) -> tuple[str | None, int]:
    """Validate a requested mode + token budget into a (system_prompt, max_new_tokens) pair.

    - An unknown/None ``mode`` falls back to ``DEFAULT_MODE`` (never raises — a bad mode
      shouldn't fail a request).
    - ``max_new_tokens`` is clamped to ``[_MIN_MAX_NEW_TOKENS, _MAX_MAX_NEW_TOKENS]``; None
      uses the mode's default. This is the "the VLM validates, never blindly trusts the
      request" half of the contract.
    """
    spec = _MODES.get(mode or DEFAULT_MODE) or _MODES[DEFAULT_MODE]
    budget = spec["default_max_new_tokens"] if max_new_tokens is None else int(max_new_tokens)
    budget = max(_MIN_MAX_NEW_TOKENS, min(_MAX_MAX_NEW_TOKENS, budget))
    return spec["system_prompt"], budget
