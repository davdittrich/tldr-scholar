"""Data models for tldr-scholar."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel


class AudienceEnum(str, Enum):
    """Audience persona for the summary."""
    EXPERT = "expert"
    LAYMAN = "layman"
    STUDENT = "student"


class ToneEnum(str, Enum):
    """Tone of the summary."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ANALYTICAL = "analytical"


class SummaryRequest(BaseModel):
    """Request parameters for summarization."""
    text: str
    max_chars: int = 500
    focus: str = "main findings and novel insights"
    hashtags: int = 0
    audience: AudienceEnum = AudienceEnum.EXPERT
    tone: ToneEnum = ToneEnum.PROFESSIONAL
    backend: Literal["auto", "gemini", "lemonade", "ollama", "extractive"] = "auto"
    backend_config: dict[str, Any] = {}


class SummaryMetadata(BaseModel):
    """Metadata about the summarization result."""
    source: str = ""
    input_type: str = ""       # "pdf", "html", "oa_pdf", "oa_full_text", "abstract", "markdown", "text"
    backend_used: str = ""
    max_chars: int = 500
    focus: str = ""
    char_count: int = 0
    audience: AudienceEnum = AudienceEnum.EXPERT
    tone: ToneEnum = ToneEnum.PROFESSIONAL
    tokens_used: int | None = None
    cost_usd: float | None = None
    cost_currency: str | None = None
    tokens_estimated: bool = False
    cost_estimated: bool = False


class SummaryResult(BaseModel):
    """Result of a summarization operation."""
    text: str
    hashtags: list[str] = []
    metadata: SummaryMetadata = SummaryMetadata()
