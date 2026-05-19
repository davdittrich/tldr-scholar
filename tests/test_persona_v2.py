"""Tests for v2 persona schema — spec test_persona_v2.py coverage.

Core behaviours from spec testing-strategy row:
  v1 file → loader raises unsupported_persona_schema       [see test_personas.py]
  v2 file → topics populated                               [see test_personas.py]
  Round-trip serialize                                     [see test_personas.py]
  Full clustered post text preserved across save/load      [HERE]
  Incomplete-persona round-trip                            [HERE]

The non-v2 counterpart test_personas.py covers the first three bullets.
This file adds the two gap-closers the spec mandates.
"""
from __future__ import annotations

from pathlib import Path

from tldr_scholar.personas import Persona, PersonaManager, write_persona_yaml


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _make_persona(
    name: str = "testpersona",
    posts: list[str] | None = None,
    status: str = "complete",
    incomplete_stages: list[str] | None = None,
) -> Persona:
    return Persona.model_validate(
        {
            "name": name,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "status": status,
            "incomplete_stages": incomplete_stages or [],
            "topics": {
                "economics+labor+wage": {
                    "label": "economics+labor+wage",
                    "centroid": [0.25] * 384,
                    "sample_size": 3,
                    "posts": posts or [],
                    "revelation_priorities": ["empirical findings"],
                    "suppression_rules": [],
                }
            },
        }
    )


# ---------------------------------------------------------------------------
# Full clustered post text preserved across save/load
# ---------------------------------------------------------------------------

def test_clustered_post_text_preserved_across_save_load(tmp_path: Path) -> None:
    """TopicProfile.posts (clustered post text) must survive write_persona_yaml + reload.

    Regression guard: ensures the 'posts' field is not silently dropped by
    yaml.safe_dump / Pydantic serialization during the save→load round-trip.
    """
    original_posts = [
        "Wages rose 3 % in Q3 but real purchasing power fell.",
        "Labor force participation among 55+ cohort declined again.",
    ]
    persona = _make_persona(name="econdave", posts=original_posts)

    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    write_persona_yaml(persona, persona_dir / "econdave.yaml")

    manager = PersonaManager(config_dir=persona_dir)
    loaded = manager.get_persona("econdave")

    assert loaded is not None, "Persona must be loadable after write_persona_yaml"
    tp = loaded.topics["economics+labor+wage"]
    assert tp.posts == original_posts, (
        f"Clustered post text not preserved. Expected {original_posts!r}, got {tp.posts!r}"
    )


# ---------------------------------------------------------------------------
# Incomplete-persona round-trip
# ---------------------------------------------------------------------------

def test_incomplete_persona_roundtrip(tmp_path: Path) -> None:
    """A Persona with status='incomplete' must round-trip through save/load intact.

    Regression guard for the partial-write path (exit code 4): the persona file
    written during LLM exhaustion must be reloadable and retain its incomplete
    status and stage list.
    """
    persona = _make_persona(
        name="partial_dave",
        status="incomplete",
        incomplete_stages=["aggregate_topic:partial"],
    )

    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    write_persona_yaml(persona, persona_dir / "partial_dave.yaml")

    manager = PersonaManager(config_dir=persona_dir)
    loaded = manager.get_persona("partial_dave")

    assert loaded is not None, "Incomplete persona must be loadable"
    assert loaded.status == "incomplete", (
        f"status must survive round-trip; got {loaded.status!r}"
    )
    assert "aggregate_topic:partial" in loaded.incomplete_stages, (
        f"incomplete_stages must survive round-trip; got {loaded.incomplete_stages!r}"
    )
