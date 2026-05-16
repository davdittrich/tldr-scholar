"""Common types and enums for tldr-scholar."""
from __future__ import annotations

from enum import Enum


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
