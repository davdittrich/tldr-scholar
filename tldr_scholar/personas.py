"""Dynamic persona management for tldr-scholar."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tldr_scholar.config import DEFAULT_PERSONA_DIR


class Persona(BaseModel):
    """Configuration for a specific writing style persona."""
    name: str
    role: str
    tone: str
    structure_pattern: str
    hashtag_style: str = "lowercase"  # "lowercase" or "pascal"
    
    # Cognitive Architecture (Deep Persona)
    agenda: Optional[str] = None
    worldview: Optional[str] = None
    revelation_priorities: list[str] = Field(default_factory=list)
    suppression_rules: list[str] = Field(default_factory=list)
    substantive_anchors: list[str] = Field(default_factory=list)
    pivot_logic: Optional[str] = None
    rhetorical_strategy: Optional[str] = None
    identifiable_nuances: list[str] = Field(default_factory=list)
    
    # Quantified Confidence (0-100)
    attribute_confidence: dict[str, int] = Field(default_factory=dict)


class PersonaManager:
    """Loads and manages personal style profiles from YAML files."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or DEFAULT_PERSONA_DIR
        self._personas: dict[str, Persona] = {}
        self._loaded = False

    def reload(self) -> None:
        """Scan config directory for persona YAML files."""
        self._personas = {}
        self._loaded = True
        if not self.config_dir.exists():
            logger.debug(f"Persona directory {self.config_dir} does not exist.")
            return

        for path in self.config_dir.glob("*.yaml"):
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    if not isinstance(data, dict):
                        logger.warning(f"Skipping invalid persona file (not a dict): {path}")
                        continue
                    # Use filename (minus extension) as persona name if not in YAML
                    name = data.get("name", path.stem)
                    data["name"] = name
                    self._personas[name] = Persona.model_validate(data)
            except Exception as e:
                logger.warning(f"Failed to load persona from {path}: {e}")
                continue

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def get_persona(self, name: str) -> Persona | None:
        """Get a persona by name."""
        self._ensure_loaded()
        return self._personas.get(name)

    def list_personas(self) -> list[str]:
        """Return list of available persona names."""
        self._ensure_loaded()
        return sorted(list(self._personas.keys()))
