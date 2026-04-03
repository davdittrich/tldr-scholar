"""Abstract base class for summarization backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BackendBase(ABC):
    """Abstract interface for summarization backends.

    All backends return a raw response string or None on failure.

    For LLM backends: the response includes summary + embedded hashtags
    (per hashtag_instruction). The caller parses them via hashtags.py.

    For the extractive backend: hashtag_instruction is ignored. Hashtags
    are generated separately by the caller via generate_hashtags_tfidf().
    """

    @abstractmethod
    def summarize(
        self,
        text: str,
        max_chars: int,
        focus: str,
        hashtag_instruction: str,
        mode: str = "scientific",
        sentence_count: int = 5,
    ) -> Optional[str]:
        """Summarize text. Returns response string or None on failure."""
