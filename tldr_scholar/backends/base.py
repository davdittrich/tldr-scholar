"""Base class for summarization backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tldr_scholar.types import AudienceEnum, ToneEnum


class BackendBase(ABC):
    """Abstract base class for all summarization backends."""

    @abstractmethod
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
        """Summarize the given text.

        Args:
            text: The text to summarize.
            max_chars: Target length in characters.
            focus: Specific topic or question to focus on.
            hashtag_instruction: Instruction for hashtag generation (empty = none).
            audience: The target audience persona.
            tone: The desired tone for the summary.
            mode: "scientific" or "general".
            sentence_count: Number of sentences in the summary.
            persona: Named persona override (e.g., "stitched").
            metadata: Additional metadata (e.g., source URL).

        Returns:
            The summary text, or None if the backend failed.
        """
        pass
