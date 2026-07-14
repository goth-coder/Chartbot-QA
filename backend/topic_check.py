"""Cheap on-topic check — is this question even about a chart?

Part of the guard latency optimization (2026-07): Llama Guard (Layer 3) used to run on
EVERY allowed request, but it does two jobs — a semantic safety net for content that
slipped past Layer 2's toxicity/injection/PII thresholds, AND the only check for
off-topic questions (its custom S99 code, "not about a chart"). This module covers the
second job cheaply, so ``guard.py`` can skip the LLM round-trip when a question is BOTH
confidently clean (Layer 2) AND confidently on-topic (here) — Layer 3 stays the fallback
for anything either check is unsure about.

**Design (embedding similarity, not zero-shot NLI):** an earlier zero-shot-classification
attempt (asking a generic NLI model "is this about a chart?") was tried and rejected —
terse real ChartQA questions like "Which year had the highest revenue?" don't mention
"chart" at all, so a bare zero-shot judgment on the text alone had a high false-negative
rate on genuine chart questions regardless of label phrasing (verified experimentally,
2026-07-13). Embedding similarity against REAL ChartQA reference questions works far
better: embed the incoming question, compare by cosine similarity to a fixed reference
set of ~80 real ChartQA training questions (below), and take the max similarity as the
on-topic score. Validated on a held-out split (60 unseen ChartQA questions + 51 varied
negatives spanning general knowledge/creative/coding/prompt-injection/PII/jailbreak
categories): threshold 0.44 was chosen to prioritize PRECISION over recall — 100%
precision (zero false "on-topic" verdicts across all 51 negatives, including every
injection/jailbreak/PII probe) at ~72% recall. A lower threshold (0.40) recovered more
recall (~87%) but let one benign off-topic question ("Who won the 2024 US election?")
through at 0.424 — not a security miss (Layer 2 still independently screens toxicity/
injection/PII regardless), but deemed not worth the risk for a UX nicety. Also ~75x
faster than the zero-shot approach (~0.04s vs ~3s for a batch).

**Lazy + fail-open**, same convention as every other guard.py detector: the model loads
on first use, and any failure (dep missing, model unavailable) returns None rather than
raising — a None result means "unsure," which guard.py treats as "call Layer 3" (fail
open toward the safety net, never toward silently skipping it).
"""
from __future__ import annotations

from functools import lru_cache

from env_config import env_bool, env_float, env_str, resolve_model_path

TOPIC_CHECK_ENABLED = env_bool("TOPIC_CHECK_ENABLED")
TOPIC_CHECK_THRESHOLD = env_float("TOPIC_CHECK_THRESHOLD")
# Local committed dir by default (backend/models/, Git LFS) — see resolve_model_path.
# all-MiniLM-L6-v2's weights are served only via HF's Xet CDN, which fails on Cloud
# Build's network, so it can't be baked from the Hub; it's loaded from disk instead.
TOPIC_CHECK_MODEL = resolve_model_path(env_str("TOPIC_CHECK_MODEL"))

# ~80 real questions from HuggingFaceM4/ChartQA's train split (random sample, seed=42),
# hardcoded so no dataset download happens at runtime — this is a fixed reference set,
# not something that needs to track the live dataset. Regenerate only if the similarity
# threshold needs recalibrating against a different/larger reference sample.
_REFERENCE_QUESTIONS = [
    "How many people visited Chester Zoo in 2019?",
    "What percentage of the municipal wastewater in Newfoundland and Labrador was discharged without treatment in 2017?",
    "What was the lowest crime clearance rate in 2019?",
    "What was the value of private equity investments in consumer goods and services?",
    "How many times was Adderall XR prescribed for children with ADHD between 10 and 19 years old?",
    "What is the first year in this data?",
    "What country has the highest maternal mortality rate among developed countries?",
    "How many vessels were registered in the United States in 2019?",
    "How much did Nigeria's real gross domestic product increase in 2019?",
    "What is the difference of low income and middle income student going to college in 2009?",
    "What was the estimated value of the Toronto Raptors in 2021?",
    "What was Trina Solar's net sales in FY 2012?",
    "How many metal bands were in Cyprus as of July 2015?",
    "How many British pounds worth of sugar was exported from the UK to the Middle East and North Africa in 2018?",
    "What was the total nondurable goods sales of U.S. merchant wholesalers in 2019?",
    "Who was the club's all-time top scorer in 2019?",
    "Which product category was the top selling during the entire period?",
    "How much is energy drinks sales more than the average sales of all drinks?",
    "Find the sum of the value between 50 to 60 in the chart.?",
    "What is the total number of Hudson's Bay Company stores in 2019?",
    "What was the female population of Botswana in 2019?",
    "What was the prison population rate in Vietnam in 2015?",
    "Which country came in third for emigration?",
    "What percentage of tweets from world leaders were in Spanish?",
    "How many road deaths were there in Belgium in 2019?",
    "Which share of believers has least value?",
    "What was Canada's last surplus in 2008?",
    "What is the average percentage of respondents who think coronavirus outbreak Highly affected the personal economy in Mexico as of May 14 and April 25?",
    "What's the value of the orange bar in Greece?",
    "How many color bars are used in the graph?",
    "How many players played in the 2016/17 season?",
    "What's the percentage of adults without health insurance who do not use internet?",
    "What was World Wrestling Entertainment's global revenue in 2020?",
    "What was the difference in net sales from 2017 to 2020 in the Americas?",
    "What was ITC Limited's gross revenue in Indian rupees in fiscal year 2020?",
    "In which the value in East Germany is highest in the Worse off or the Better off?",
    "What's the difference in the value of Portugal and Iceland?",
    "What was the share of gross advances for first time buyers in the fourth quarter of 2020?",
    "What is the best selling car brand in Canada?",
    "What is the average of the two bars?",
    "What was the most studied language in Italian schools in 2019?",
    "What was the national unemployment rate in 2020?",
    "What was Hawaii's unemployment rate in 2020?",
    "What was the box office revenue in Canada in the last measured period?",
    "Average the minimum opinion percentages of the three scenarios?",
    "What was the average expenditure on curtains and draperies per consumer unit in the United States in 2019?",
    "What was the second most popular app in the Google Play Store?",
    "How many people watched the Sochi Winter Olympics in Africa?",
    "What was the value of Axis Bank's gross non-performing assets in Indian rupees in fiscal year 2020?",
    "Which state sold about 3,670 BEVs in 2016?",
    "What's the sum of all the yellow bars above 50?",
    "What was the number of companies operating in the Bulgarian insurance market from 2011 to 2014?",
    "WHat does grey indicate?",
    "Which country has the highest sales revenue?",
    "How much revenue did Weatherford generate in 2020?",
    "What is the total share of respondents avoided large public assemblings?",
    "What was Ryanair's net profit in 2017/18?",
    "How many people in the metropolitan region of Buenos Aires watched TV on March 2, 2020?",
    "How much did Spain's wine export volume grow during the period considered?",
    "For how many years is the value of Spain lower than that of Marshall Islands?",
    "What is the average life expectancy in 2017 and 2018 (combined)?",
    "What was the infant mortality rate in Paraguay in 2019?",
    "How many viewers watched the game between Michigan State and Ohio State on average?",
    "What was Aruba's internal consumption of travel and tourism in dollars in 2019?",
    "What percentage of total net sales did Prada's footwear product line account for in 2020?",
    "Where was the highest value of goods exported to?",
    "Which company was the top smartwatch company in 2013?",
    "When was the BFI London Film Festival first held?",
    "What percentage of respondents stated they were between 13 and 16 years old when they started working as a model?",
    "How much of Dow's revenue did it generate in the United States in 2020?",
    "What was Infineon's market share in the global microcontroller-based chip card ICs market in 2018?",
    "What is the total percentage of people who favor legalizing marijuana?",
    "Which consists of half of the pie chart?",
    "Find out the average of the bottom two countries ??",
    "How many of Finland's hotels are seasonally open?",
    "What is the market size of the smart kitchen market expected to reach by 2027?",
    "Which candidate had the highest percentage of votes after the second round of the Ukrainian presidential elections in 2019?",
    "What was England's gross domestic product in 2019?",
    "Work out the ratio between the least tend to favor one side percent and the most deal fairly with all sides percent",
    "What is the value for Ron Santo?",
]


@lru_cache(maxsize=1)
def _load_model():
    """Load the embedding model + tokenizer once. Returns (tokenizer, model) or None."""
    try:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOPIC_CHECK_MODEL)
        model = AutoModel.from_pretrained(TOPIC_CHECK_MODEL)
        model.eval()
        return tok, model
    except Exception:  # noqa: BLE001 — dep missing or model download failed
        return None


def _embed(texts: list[str]):
    """Mean-pooled, L2-normalized sentence embeddings for a batch of texts."""
    import torch

    tok, model = _load_model()
    inputs = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(summed / counts, dim=1)


@lru_cache(maxsize=1)
def _reference_embeddings():
    """Embeddings of the fixed reference question set, computed once and cached."""
    return _embed(_REFERENCE_QUESTIONS)


def on_topic_confidence(question: str) -> float | None:
    """Max cosine similarity to the reference set, in [-1,1] (in practice [0,1] for
    real text), or None if unavailable.

    A high score means confidently on-topic (similar to a real ChartQA question); a
    low/mid score does NOT mean "off-topic for sure" — it means "not confident," which
    the caller should treat as ambiguous.
    """
    if not TOPIC_CHECK_ENABLED:
        return None
    if _load_model() is None:
        return None
    try:
        q_emb = _embed([question])
        ref_emb = _reference_embeddings()
        sims = q_emb @ ref_emb.T
        return float(sims.max())
    except Exception:  # noqa: BLE001
        return None


def is_confidently_on_topic(question: str) -> bool:
    """True only when the embedder ran AND scored above TOPIC_CHECK_THRESHOLD.

    Unavailable/unsure both return False — the caller's fail-open direction is "call
    Layer 3", not "assume on-topic".
    """
    score = on_topic_confidence(question)
    return score is not None and score >= TOPIC_CHECK_THRESHOLD


def warmup() -> None:
    """Pre-load the model + reference embeddings off the request path (boot, in a thread)."""
    if _load_model() is not None:
        _reference_embeddings()


def is_available() -> bool:
    """Whether the embedding model is actually loaded (for /api/health, debugging)."""
    return _load_model() is not None
