"""Scrape-time injection filter for tldr-scholar.

Detects likely prompt-injection attempts in scraped content before
it enters the clustering pipeline. Matched posts are dropped pre-cluster;
a warn envelope is emitted (source only, never matched content).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from loguru import logger

# ---------------------------------------------------------------------------
# Pattern loading
# ---------------------------------------------------------------------------

_PATTERNS_FILE = Path(__file__).parent.parent / "tests" / "fixtures" / "injection_patterns.txt"

_COMPILED: list[re.Pattern] | None = None


def _load_patterns() -> list[re.Pattern]:
    """Load and compile patterns from injection_patterns.txt. Cached after first call."""
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED

    patterns: list[re.Pattern] = []
    try:
        text = _PATTERNS_FILE.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                patterns.append(re.compile(stripped, re.IGNORECASE))
            except re.error as exc:
                logger.warning(f"scrape_filter: skipping invalid pattern {stripped!r}: {exc}")
    except FileNotFoundError:
        logger.warning(f"scrape_filter: patterns file not found at {_PATTERNS_FILE}")

    _COMPILED = patterns
    return _COMPILED


# ---------------------------------------------------------------------------
# Zero-width / control character stripping
# ---------------------------------------------------------------------------

_ZERO_WIDTH = frozenset(["​", "‌", "‍", "﻿"])


def _normalize(text: str) -> str:
    """NFKC-normalize + strip zero-width + strip ASCII control characters."""
    # 1. NFKC normalization (resolves fullwidth, ligatures, etc.)
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Remove zero-width chars — replace with space to preserve word boundaries
    # (e.g. "ignore​previous" → "ignore previous" so patterns still match)
    normalized = "".join(" " if ch in _ZERO_WIDTH else ch for ch in normalized)

    # 3. Replace ASCII control characters (0x00–0x1F, 0x7F) with a space, except tab/LF/CR
    # Using space (not removal) preserves word boundaries that injectors hide by inserting
    # control chars between words (e.g. "ignore\x01previous" → "ignore previous").
    normalized = "".join(
        ch if ch in ("\t", "\n", "\r") or (ord(ch) > 0x1F and ord(ch) != 0x7F)
        else " "
        for ch in normalized
    )

    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_likely_injection(text: str) -> bool:
    """Return True if `text` matches any known injection pattern.

    Pipeline:
    1. NFKC normalize + strip zero-width / control chars.
    2. Regex scan normalized form against injection_patterns.txt.

    Matched content is NEVER echoed in logs — only the boolean result is returned.
    """
    normalized = _normalize(text)
    patterns = _load_patterns()
    for pattern in patterns:
        if pattern.search(normalized):
            return True
    return False
