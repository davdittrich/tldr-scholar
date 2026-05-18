"""Dynamic persona management for tldr-scholar (v2 schema)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from tldr_scholar.config import DEFAULT_PERSONA_DIR


# ---------------------------------------------------------------------------
# Forward-declared helpers (avoid circular import; error_contract imported lazily)
# ---------------------------------------------------------------------------

def _warn_if_incomplete(persona: "Persona") -> None:
    """Emit a warn envelope if the persona has status='incomplete'.

    Imported lazily to avoid circular dependency between personas and error_contract.
    """
    if persona.status == "incomplete":
        from tldr_scholar.error_contract import emit_envelope  # noqa: PLC0415
        emit_envelope(
            level="warn",
            stage="load",
            code="persona_incomplete",
            message=(
                f"Persona '{persona.name}' has status=incomplete "
                f"(failed stages: {persona.incomplete_stages}). "
                "Generation results may be degraded."
            ),
        )


def write_persona_yaml(persona: "Persona", path: Path) -> None:
    """Serialize *persona* to *path* as YAML (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(persona.model_dump(), sort_keys=False))


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------

class TopicProfile(BaseModel):
    """Per-topic emphasis profile derived from clustered post deltas."""
    label: str
    centroid: list[float]
    sample_size: int
    posts: list[str] = Field(default_factory=list)
    revelation_priorities: list[str] = Field(default_factory=list)
    suppression_rules: list[str] = Field(default_factory=list)
    substantive_anchors: list[str] = Field(default_factory=list)
    rhetorical_strategy: str = ""
    confidence: dict[str, float] = Field(default_factory=dict)


class DeltaRecord(BaseModel):
    """Per-baseline correlation record between a post and source statements."""
    baseline_type: Literal["claims", "extractive", "abstractive"]
    statements: list[str]
    status_per_statement: list[Literal["shared", "suppressed", "distorted"]]
    intent: str | None = None


class Persona(BaseModel):
    """Configuration for a writing style persona (v2 schema)."""
    name: str
    embedding_model: str  # mandatory in v2; e.g. "sentence-transformers/all-MiniLM-L6-v2"
    status: Literal["complete", "incomplete"] = "complete"
    incomplete_stages: list[str] = Field(default_factory=list)

    # Global synthesis fields (populated by DEEP_SYNTHESIS_PROMPT)
    agenda: str = ""
    worldview: str = ""
    pivot_logic: str = ""
    identifiable_nuances: list[str] = Field(default_factory=list)
    attribute_confidence: dict[str, int] = Field(default_factory=dict)

    # Per-topic profiles; mandatory — min 1 topic (fallback "_global")
    topics: dict[str, TopicProfile]

    # Optional persona display fields
    role: str = ""
    tone: str = ""
    structure_pattern: str = ""
    hashtag_style: str = "lowercase"


# ---------------------------------------------------------------------------
# Persona manager
# ---------------------------------------------------------------------------

def _is_v1_shape(data: dict) -> bool:
    """Return True if `data` looks like a v1 Persona (top-level v1-only fields, no topics/embedding_model)."""
    v1_fields = {"revelation_priorities", "suppression_rules", "substantive_anchors", "rhetorical_strategy"}
    has_v1 = bool(v1_fields & set(data.keys()))
    missing_v2 = "topics" not in data or "embedding_model" not in data
    return has_v1 and missing_v2


class PersonaManager:
    """Loads and manages personal style profiles from YAML files."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or DEFAULT_PERSONA_DIR
        self._personas: dict[str, Persona] = {}
        self._loaded = False

    def reload(self) -> None:
        """Scan config directory for persona YAML files.

        Raises:
            ValidationError: If a file fails Pydantic v2 schema validation (not swallowed).
            SystemExit(2): If a v1-shape file is detected (unsupported_persona_schema).
        """
        self._personas = {}
        self._loaded = True
        if not self.config_dir.exists():
            logger.debug(f"Persona directory {self.config_dir} does not exist.")
            return

        for path in self.config_dir.glob("*.yaml"):
            try:
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                logger.warning(f"Skipping malformed YAML file {path.name}: {exc}")
                continue

            if not isinstance(data, dict):
                logger.warning(f"Skipping invalid persona file (not a dict): {path}")
                continue

            if _is_v1_shape(data):
                logger.error(
                    '{"level":"error","stage":"load","code":"unsupported_persona_schema",'
                    f'"message":"v1-shape persona file detected: {path.name}. '
                    'Delete and re-derive with tldr-scholar-synthesize-style."}'
                )
                sys.exit(2)

            # Use filename (minus extension) as persona name if not in YAML
            name = data.get("name", path.stem)
            data["name"] = name
            # ValidationError propagates — no swallowing
            self._personas[name] = Persona.model_validate(data)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def get_persona(self, name: str) -> Persona | None:
        """Get a persona by name."""
        self._ensure_loaded()
        persona = self._personas.get(name)
        if persona is not None:
            _warn_if_incomplete(persona)
        return persona

    def list_personas(self) -> list[str]:
        """Return list of available persona names."""
        self._ensure_loaded()
        return sorted(list(self._personas.keys()))
