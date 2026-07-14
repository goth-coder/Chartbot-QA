"""Layer-2 input guard: small local classifiers (see docs/PLAN.md §6).

Three independent signals, each its own small encoder model:
  - toxicity        (detoxify / unitary toxic-bert)
  - prompt injection (protectai deberta-v3 prompt-injection)
  - PII             (Microsoft Presidio)

Design constraints:
  * **Lazy + fail-open.** Each detector imports its (heavy) ML dependency only on
    first use and returns ``None`` if the dependency or model isn't available, so
    the backend still boots and serves in dev without torch/transformers/presidio.
    Install the real models with:  pip install -r backend/requirements-guard.txt
  * **Detectors are module-level callables** so tests can monkeypatch them with no
    ML deps installed.
  * Runs AFTER Layer 1 (cheap rules in app.py), BEFORE the model. The guard never
    sees an image here — it screens the question text.

Contract: ``guard(question) -> GuardResult``. Fail-closed for safety categories
(toxic / prompt_injection); PII blocks only on high-risk entity types.

**Layer 3 is now CONDITIONAL, not unconditional (2026-07 latency fix).** Llama Guard
(guard_llm.llm_classify) does two jobs: a semantic safety net for content that scored
low-but-not-zero on Layer 2, and the only check for off-topic questions. Calling it on
every single allowed request (even an obviously clean "what was the peak value?") was
the dominant latency cost of the whole guard stack. It's now skipped ONLY when a
question is BOTH confidently clean on every Layer-2 signal (below the LOW thresholds
below, not just under the block thresholds) AND confidently on-topic per
``topic_check.py`` — see ``guard()`` for the exact cascade. Anything either check is
unsure about still goes to Layer 3, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from env_config import env_bool, env_float, env_str

# --- Tunables (required in .env; no in-code defaults) -----------------------
TOXICITY_THRESHOLD = env_float("GUARD_TOXICITY_THRESHOLD")
INJECTION_THRESHOLD = env_float("GUARD_INJECTION_THRESHOLD")
PII_SCORE_THRESHOLD = env_float("GUARD_PII_THRESHOLD")
GUARD_ENABLED = env_bool("GUARD_ENABLED")
# Model identifiers (swap without code changes, e.g. a smaller/quantized variant).
TOXICITY_MODEL = env_str("GUARD_TOXICITY_MODEL")
INJECTION_MODEL = env_str("GUARD_INJECTION_MODEL")
# "Confidently clean" — well BELOW the block thresholds above. A score between this and
# the block threshold is the ambiguous band Llama Guard is actually good at judging; only
# a score below BOTH low thresholds (plus no PII hits) is confident enough to skip it.
TOXICITY_LOW = env_float("GUARD_TOXICITY_LOW")
INJECTION_LOW = env_float("GUARD_INJECTION_LOW")

# Block only on high-risk identifiers; ignore PERSON/LOCATION/ORG/DATE to avoid
# false positives on ordinary chart questions ("What was John's revenue?").
_SENSITIVE_PII = {
    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "CRYPTO",
    "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "IP_ADDRESS", "MEDICAL_LICENSE",
}


@dataclass
class GuardResult:
    allowed: bool
    category: str = "ok"      # "ok" | "toxic" | "prompt_injection" | "pii"
    reason: str = ""          # short, user-safe message


# --- Device selection (use CUDA when available, else CPU) -------------------
@lru_cache(maxsize=1)
def _cuda_device_index() -> int:
    """0 for the first CUDA GPU, -1 for CPU (transformers `device` convention)."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:  # noqa: BLE001 — torch not installed
        return -1


# --- Model loaders (cached; return None if unavailable) ---------------------
@lru_cache(maxsize=1)
def _load_toxicity():
    try:
        from detoxify import Detoxify
        device = "cuda" if _cuda_device_index() == 0 else "cpu"
        return Detoxify(TOXICITY_MODEL, device=device)
    except Exception:  # noqa: BLE001 — dep missing or model download failed
        return None


@lru_cache(maxsize=1)
def _load_injection():
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model=INJECTION_MODEL,
            truncation=True,
            device=_cuda_device_index(),
        )
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def _load_pii():
    try:
        from presidio_analyzer import AnalyzerEngine
        return AnalyzerEngine()
    except Exception:  # noqa: BLE001
        return None


# --- Detectors (return a score / entities, or None if unavailable) ----------
def toxicity_score(text: str) -> float | None:
    """P(toxic) in [0,1], or None if the detector isn't available."""
    model = _load_toxicity()
    if model is None:
        return None
    try:
        return float(model.predict(text)["toxicity"])
    except Exception:  # noqa: BLE001
        return None


def injection_score(text: str) -> float | None:
    """P(prompt injection) in [0,1], or None if unavailable."""
    clf = _load_injection()
    if clf is None:
        return None
    try:
        out = clf(text)[0]
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        # Model returns the winning label + its confidence; convert to P(INJECTION).
        return score if label == "INJECTION" else 1.0 - score
    except Exception:  # noqa: BLE001
        return None


def pii_hits(text: str) -> list[str] | None:
    """High-risk PII entity types found, or None if the detector isn't available."""
    engine = _load_pii()
    if engine is None:
        return None
    try:
        results = engine.analyze(text=text, language="en")
        return sorted({
            r.entity_type for r in results
            if r.entity_type in _SENSITIVE_PII and r.score >= PII_SCORE_THRESHOLD
        })
    except Exception:  # noqa: BLE001
        return None


# --- Orchestrator -----------------------------------------------------------
def guard(question: str) -> GuardResult:
    """Screen the question through the Layer-2 classifiers, then — CONDITIONALLY — the
    Layer-3 LLM.

    Returns ``allowed=True`` when nothing fires OR when a detector is unavailable
    (fail-open). Order: toxicity -> prompt injection -> PII (any of these BLOCKS
    outright, unchanged) -> confidence check -> Layer-3 LLM, but the LLM call is now
    SKIPPED when the question is both confidently clean (every Layer-2 score below its
    LOW threshold) and confidently on-topic (topic_check.py). Anything either check is
    unsure about still reaches Layer 3, same as before this optimization.
    """
    if not GUARD_ENABLED:
        return GuardResult(True)

    tox = toxicity_score(question)
    if tox is not None and tox >= TOXICITY_THRESHOLD:
        return GuardResult(False, "toxic", "This question looks abusive — please rephrase.")

    inj = injection_score(question)
    if inj is not None and inj >= INJECTION_THRESHOLD:
        return GuardResult(
            False, "prompt_injection",
            "That looks like an attempt to override the assistant's instructions.",
        )

    hits = pii_hits(question)
    if hits:
        return GuardResult(
            False, "pii",
            "Please remove personal data from your question (e.g. " + ", ".join(hits) + ").",
        )

    # Confidently clean on Layer 2 = every score present AND below its LOW threshold (a
    # detector that's unavailable, i.e. None, does NOT count as "confidently clean" — a
    # missing signal is exactly the kind of uncertainty that should still reach Layer 3).
    layer2_confident_clean = (
        tox is not None and tox < TOXICITY_LOW
        and inj is not None and inj < INJECTION_LOW
        and hits is not None  # ran and found nothing (hits == [] falls through above)
    )

    if layer2_confident_clean:
        from topic_check import is_confidently_on_topic
        if is_confidently_on_topic(question):
            # Both cheap checks agree: clean AND on-topic — skip the LLM round-trip.
            try:
                import metrics
                metrics.count_guard_layer3_skipped()
            except Exception:  # noqa: BLE001 — metrics are themselves fail-open
                pass
            return GuardResult(True)

    # --- Layer 3: LLM input boundary filter (Llama Guard) — gated, fail-open ---
    # Reached whenever Layer 2 is anything less than confidently clean, OR the topic
    # check didn't confirm on-topic (including when the classifier is unavailable).
    # Lazy import avoids a circular import (guard_llm imports GuardResult from here).
    try:
        from guard_llm import llm_classify
        verdict = llm_classify(question)
    except Exception:  # noqa: BLE001
        verdict = None
    if verdict is not None and not verdict.allowed:
        return verdict

    return GuardResult(True)


def warmup() -> None:
    """Pre-load every guard model off the request path (call at boot, in a thread).

    Fail-open: if a model's deps aren't installed the loader just returns None and this
    does nothing heavy. Loads Layer-2 encoders + the topic-check classifier, then warms
    the Layer-3 LLM (still warmed unconditionally — it's the fallback for anything the
    cheap checks are unsure about, so it must be ready regardless of how often it ends
    up actually being called).
    """
    _load_toxicity()
    _load_injection()
    _load_pii()
    try:
        import topic_check
        topic_check.warmup()
    except Exception:  # noqa: BLE001
        pass
    try:
        import guard_llm
        guard_llm.warmup()
    except Exception:  # noqa: BLE001
        pass


def is_available() -> dict[str, bool]:
    """Which detectors actually have their model loaded (for /api/health, debugging)."""
    try:
        import topic_check
        topic_available = topic_check.is_available()
    except Exception:  # noqa: BLE001
        topic_available = False
    return {
        "toxicity": _load_toxicity() is not None,
        "prompt_injection": _load_injection() is not None,
        "pii": _load_pii() is not None,
        "topic_check": topic_available,
    }
