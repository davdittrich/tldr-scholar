"""Gemini backend via shared gemini-acp package."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from tldr_scholar.backends.base import BackendBase
from tldr_scholar.config import GeminiConfig
from tldr_scholar.prompts import PromptBuilder

if TYPE_CHECKING:
    from tldr_scholar.models import AudienceEnum, ToneEnum

try:
    from gemini_acp import ACP_AVAILABLE, summarize_via_gemini
except ImportError:
    summarize_via_gemini = None  # type: ignore[assignment]
    ACP_AVAILABLE = False


class GeminiBackend(BackendBase):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self._model = cfg.get("model", "")
        self._timeout = cfg.get("timeout", GeminiConfig().timeout)
        self._last_usage = None

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
    ) -> Optional[str]:
        if summarize_via_gemini is None:
            logger.debug("gemini-acp not installed")
            return None

        if not ACP_AVAILABLE:
            logger.debug("ACP library not available")
            return None

        prompt = PromptBuilder().build_single_prompt(
            text=text,
            mode=mode,
            max_chars=max_chars,
            focus=focus,
            hashtag_instruction=hashtag_instruction,
            sentence_count=sentence_count,
            audience=audience,
            tone=tone,
        )
        text_result, usage = summarize_via_gemini(
            text="",  # text already embedded in prompt via <document> delimiters
            prompt=prompt,
            model=self._model,
            timeout=self._timeout,
        )
        self._last_usage = usage
        return text_result
