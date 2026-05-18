"""Correlates a social media post against all populated source baselines (WU-3).

For each populated baseline in SourceBaselines, builds a DeltaRecord via
CORRELATION_PROMPT. Returns a list with one DeltaRecord per successful
correlation; empty list if all fail or all baselines are None.
"""
from __future__ import annotations

import re
import yaml
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from tldr_scholar.personas import DeltaRecord
from tldr_scholar.prompts import CORRELATION_PROMPT
from tldr_scholar.source_baseline import SourceBaselines

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
LLMCaller = Callable[[str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Sentence splitter (stdlib — no NLTK required)
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* on sentence boundaries. Returns 1-item list if no split found."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts if parts else [text.strip()]


# ---------------------------------------------------------------------------
# Correlation YAML parser
# ---------------------------------------------------------------------------
_VALID_STATUSES = {"shared", "suppressed", "distorted"}


def _parse_correlation_yaml(raw: Any) -> Optional[list[dict[str, Any]]]:
    """Parse CORRELATION_PROMPT response into a list of {status, intent} dicts."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].strip() if len(parts) > 1 else text
        if text.startswith("yaml\n"):
            text = text[5:]
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return parsed if isinstance(parsed, list) else None


def _extract_status_and_intent(
    corr_items: list[dict[str, Any]],
    statements: list[str],
) -> tuple[list[str], Optional[str]]:
    """Derive status_per_statement and overall intent from correlation items.

    Aligns correlation rows to statements by index (or statement_id).
    Falls back to 'shared' for unmatched rows to avoid silent data loss.
    """
    n = len(statements)
    status_per_statement: list[str] = []
    intents: list[str] = []

    for idx in range(n):
        # Try to find matching item by index
        item = corr_items[idx] if idx < len(corr_items) else {}
        raw_status = str(item.get("status", "shared")).lower().strip()
        status = raw_status if raw_status in _VALID_STATUSES else "shared"
        status_per_statement.append(status)
        intent_val = item.get("intent")
        if intent_val:
            intents.append(str(intent_val))

    intent = " ".join(intents) if intents else None
    return status_per_statement, intent


# ---------------------------------------------------------------------------
# Per-baseline correlation helper
# ---------------------------------------------------------------------------


async def _correlate_one(
    baseline_type: str,
    statements: list[str],
    post_text: str,
    llm_call: LLMCaller,
) -> Optional[DeltaRecord]:
    """Correlate post against *statements* for a single baseline type.

    Returns DeltaRecord on success, None on failure.
    """
    statements_yaml = yaml.dump(statements)
    prompt = CORRELATION_PROMPT.format(statements=statements_yaml, post_text=post_text)
    try:
        raw = await llm_call(prompt)
    except Exception as exc:
        logger.warning(f"correlator: {baseline_type} LLM call failed — {exc}")
        return None

    items = _parse_correlation_yaml(raw)
    if items is None:
        logger.warning(f"correlator: {baseline_type} parse failed (bad YAML)")
        return None

    # Pad items if LLM returned fewer rows than statements
    while len(items) < len(statements):
        items.append({"status": "shared", "intent": None})

    status_per_statement, intent = _extract_status_and_intent(items, statements)

    try:
        return DeltaRecord(
            baseline_type=baseline_type,  # type: ignore[arg-type]
            statements=statements,
            status_per_statement=status_per_statement,  # type: ignore[arg-type]
            intent=intent,
        )
    except Exception as exc:
        logger.warning(f"correlator: DeltaRecord construction failed for {baseline_type} — {exc}")
        return None


# ---------------------------------------------------------------------------
# Public coroutine
# ---------------------------------------------------------------------------


async def correlate_against_baselines(
    post_text: str,
    baselines: SourceBaselines,
    llm_call: Optional[LLMCaller] = None,
) -> list[DeltaRecord]:
    """Produce one DeltaRecord per populated baseline.

    Parameters
    ----------
    post_text:
        Raw social-media post text.
    baselines:
        SourceBaselines as produced by build_baselines().
    llm_call:
        Async LLM caller (injected for tests).

    Returns
    -------
    List of DeltaRecord. Empty if all baselines are None or all correlations fail.
    """
    records: list[DeltaRecord] = []

    # --- claims ---------------------------------------------------------------
    if baselines.claims is not None:
        rec = await _correlate_one("claims", baselines.claims, post_text, llm_call)  # type: ignore[arg-type]
        if rec is not None:
            records.append(rec)
        else:
            logger.warning("correlator: claims correlation failed, dropping record")

    # --- extractive -----------------------------------------------------------
    if baselines.extractive_summary is not None:
        sentences = _split_sentences(baselines.extractive_summary)
        rec = await _correlate_one("extractive", sentences, post_text, llm_call)  # type: ignore[arg-type]
        if rec is not None:
            records.append(rec)
        else:
            logger.warning("correlator: extractive correlation failed, dropping record")

    # --- abstractive ----------------------------------------------------------
    if baselines.abstractive_summary is not None:
        sentences = _split_sentences(baselines.abstractive_summary)
        rec = await _correlate_one("abstractive", sentences, post_text, llm_call)  # type: ignore[arg-type]
        if rec is not None:
            records.append(rec)
        else:
            logger.warning("correlator: abstractive correlation failed, dropping record")

    return records
