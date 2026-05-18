"""3-baseline source decomposition for the NWL delta pipeline (WU-3).

Baselines:
  claims       — atomic claims via DECOMPOSITION_PROMPT + LLM
  extractive   — LexRank 5-sentence summary (CPU-only, deterministic)
  abstractive  — 3-sentence neutral summary via NEUTRAL_SUMMARY_PROMPT + LLM

full=False (default) → claims only; extractive and abstractive skipped.
full=True            → all three attempted; each failure isolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import yaml
from loguru import logger

from tldr_scholar.prompts import DECOMPOSITION_PROMPT, NEUTRAL_SUMMARY_PROMPT

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
LLMCaller = Callable[[str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Sumy extractive helper (CPU-only, no LLM)
# ---------------------------------------------------------------------------
_SUMY_SENTENCE_COUNT = 5


def _extractive_summarize(text: str) -> str:
    """Run LexRank over *text* and return up to _SUMY_SENTENCE_COUNT sentences joined."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sentences = summarizer(parser.document, _SUMY_SENTENCE_COUNT)
    return " ".join(str(s) for s in sentences)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SourceBaselines:
    """Holds all three baseline representations for one source."""

    claims: Optional[list[str]]             # atomic claim strings (None if failed)
    extractive_summary: Optional[str]       # LexRank joined sentences (None if failed)
    abstractive_summary: Optional[str]      # Gemini neutral summary (None if failed)


# ---------------------------------------------------------------------------
# Claims builder (shared between default and full mode)
# ---------------------------------------------------------------------------


def _parse_claims_yaml(raw: Any) -> Optional[list[str]]:
    """Parse LLM YAML response into a list of claim strings. Returns None on bad shape."""
    if not raw or not isinstance(raw, str):
        return None
    # Strip Markdown fences
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

    if not isinstance(parsed, list):
        return None

    claims: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            val = item.get("claim") or item.get("content") or item.get("text")
            if val:
                claims.append(str(val))
        elif isinstance(item, str):
            claims.append(item)

    return claims if claims else None


# ---------------------------------------------------------------------------
# Main public coroutine
# ---------------------------------------------------------------------------


async def build_baselines(
    source_text: str,
    full: bool = False,
    llm_call: Optional[LLMCaller] = None,
) -> SourceBaselines:
    """Produce baselines for one source.

    Parameters
    ----------
    source_text:
        Raw source document text.
    full:
        False (default) → claims only.
        True → claims + extractive (sumy) + abstractive (LLM).
    llm_call:
        Async callable that accepts a prompt string and returns the LLM response.
        Injected for testability; must not be None when claims/abstractive run.
    """
    # --- claims baseline (always attempted) ---------------------------------
    claims: Optional[list[str]] = None
    try:
        prompt = DECOMPOSITION_PROMPT.format(text=source_text)
        raw = await llm_call(prompt)  # type: ignore[misc]
        claims = _parse_claims_yaml(raw)
        if claims is None:
            logger.warning("source_baseline: claims parse failed (empty or bad YAML)")
    except Exception as exc:
        logger.warning(f"source_baseline: claims LLM failed — {exc}")

    if not full:
        return SourceBaselines(claims=claims, extractive_summary=None, abstractive_summary=None)

    # --- extractive baseline (CPU-only) -------------------------------------
    extractive_summary: Optional[str] = None
    try:
        extractive_summary = _extractive_summarize(source_text) or None
    except Exception as exc:
        logger.warning(f"source_baseline: extractive (sumy) failed — {exc}")

    # --- abstractive baseline (LLM) -----------------------------------------
    abstractive_summary: Optional[str] = None
    try:
        prompt = NEUTRAL_SUMMARY_PROMPT.format(source_text=source_text)
        raw = await llm_call(prompt)  # type: ignore[misc]
        abstractive_summary = raw.strip() if isinstance(raw, str) and raw.strip() else None
    except Exception as exc:
        logger.warning(f"source_baseline: abstractive LLM failed — {exc}")

    return SourceBaselines(
        claims=claims,
        extractive_summary=extractive_summary,
        abstractive_summary=abstractive_summary,
    )
