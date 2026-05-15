"""Ollama backend — /api/generate endpoint."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import httpx
from loguru import logger

from tldr_scholar.backends.base import BackendBase
from tldr_scholar.prompts import PromptBuilder

if TYPE_CHECKING:
    from tldr_scholar.models import AudienceEnum, ToneEnum


class OllamaBackend(BackendBase):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self._model = cfg.get("model", "gemma3:9b")
        self._host = cfg.get("host", "http://localhost:11434")
        self._timeout = cfg.get("timeout", 90)

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
        try:
            response = httpx.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            return response.json()["response"].strip() or None
        except Exception as e:
            logger.debug(f"Ollama request failed: {e}")
            return None
