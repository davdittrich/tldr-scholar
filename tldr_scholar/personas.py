"""Dynamic persona management for tldr-scholar."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


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
        if config_dir is None:
            self.config_dir = Path.home() / ".config" / "tldr-scholar" / "personas"
        else:
            self.config_dir = config_dir
        
        self._personas: dict[str, Persona] = {}
        self.reload()

    def reload(self) -> None:
        """Scan config directory for persona YAML files."""
        self._personas = {}
        if not self.config_dir.exists():
            return

        for path in self.config_dir.glob("*.yaml"):
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    if not isinstance(data, dict):
                        continue
                    # Use filename (minus extension) as persona name if not in YAML
                    name = data.get("name", path.stem)
                    data["name"] = name
                    self._personas[name] = Persona.model_validate(data)
            except Exception:
                # Skip invalid YAML files
                continue

    def get_persona(self, name: str) -> Persona | None:
        """Get a persona by name."""
        return self._personas.get(name)

    def list_personas(self) -> list[str]:
        """Return list of available persona names."""
        return sorted(list(self._personas.keys()))
