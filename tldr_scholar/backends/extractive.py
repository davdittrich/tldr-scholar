"""Extractive summarization backend using LexRank."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

from tldr_scholar.backends.base import BackendBase

if TYPE_CHECKING:
    from tldr_scholar.models import AudienceEnum, ToneEnum


class ExtractiveBackend(BackendBase):
    """Summarizer using traditional extractive LexRank algorithm."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}

    def summarize(
        self,
        text: str,
        max_chars: int,
        focus: str,
        hashtag_instruction: str,
        audience: AudienceEnum,
        tone: ToneEnum,
        mode: str = "scientific",
        sentence_count: int = 5,
        persona: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Optional[str]:
        """Summarize via LexRank (extractive).

        Ignores hashtag_instruction, audience, tone, and persona (limitations of extractive).
        Uses focus keywords for simple biasing if provided.
        """
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()

            # Bias towards focus if present
            if focus:
                # LexRank implementation in sumy doesn't directly support focus
                # but we can simulate it by increasing sentence count then filtering
                count = max(sentence_count * 2, 10)
            else:
                count = sentence_count

            sentences = summarizer(parser.document, count)

            # Convert to strings
            results = [str(s) for m in sentences for s in (m,) if str(s).strip()]

            if focus:
                # Naive keyword search to prioritize relevant sentences
                keywords = focus.lower().split()
                results.sort(key=lambda s: sum(1 for k in keywords if k in s.lower()), reverse=True)
                results = results[:sentence_count]

            # Join and truncate to max_chars
            full_text = " ".join(results)
            while len(full_text) > max_chars and len(results) > 1:
                results.pop()
                full_text = " ".join(results)

            if len(full_text) > max_chars:
                full_text = full_text[: max_chars - 1] + "\u2026"

            return full_text
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            return None
