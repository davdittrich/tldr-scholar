"""Tests for dynamic persona infrastructure (v2 schema)."""
from __future__ import annotations

import sys
import yaml
import pytest
from pathlib import Path
from pydantic import ValidationError

from tldr_scholar.personas import PersonaManager, Persona, TopicProfile, DeltaRecord


def _v2_persona_data(name: str = "testpersona") -> dict:
    """Minimal valid v2 persona data."""
    return {
        "name": name,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "role": "academic economist",
        "tone": "analytical",
        "structure_pattern": "stitched",
        "hashtag_style": "pascal",
        "topics": {
            "economics+labor+wage": {
                "label": "economics+labor+wage",
                "centroid": [0.1] * 384,
                "sample_size": 42,
                "revelation_priorities": ["empirical findings"],
                "suppression_rules": ["noise"],
            }
        },
        "attribute_confidence": {"role": 100, "topics": 90},
    }


def test_persona_v2_roundtrip():
    """v2 Persona loads and round-trips fields correctly."""
    data = _v2_persona_data()
    p = Persona.model_validate(data)
    assert p.name == "testpersona"
    assert p.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert p.status == "complete"
    assert p.incomplete_stages == []
    assert "economics+labor+wage" in p.topics
    tp = p.topics["economics+labor+wage"]
    assert tp.revelation_priorities == ["empirical findings"]
    assert tp.centroid[0] == pytest.approx(0.1)


def test_persona_manager_loading_v2(tmp_path):
    """PersonaManager loads v2 YAML file correctly."""
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()

    data = _v2_persona_data("davdittrich")
    with open(persona_dir / "davdittrich.yaml", "w") as f:
        yaml.dump(data, f)

    manager = PersonaManager(config_dir=persona_dir)
    assert "davdittrich" in manager.list_personas()

    p = manager.get_persona("davdittrich")
    assert p.role == "academic economist"
    assert p.hashtag_style == "pascal"
    assert p.attribute_confidence["role"] == 100
    assert "economics+labor+wage" in p.topics


def test_persona_manager_empty_dir(tmp_path):
    manager = PersonaManager(config_dir=tmp_path / "nonexistent")
    assert manager.list_personas() == []


def test_persona_manager_invalid_yaml(tmp_path):
    """Non-dict YAML is skipped (logged warning, not raised)."""
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    with open(persona_dir / "invalid.yaml", "w") as f:
        f.write("not a: dictionary: structure")

    manager = PersonaManager(config_dir=persona_dir)
    assert manager.list_personas() == []


def test_v1_shape_raises_system_exit_2(tmp_path):
    """v1-shape file (revelation_priorities at top-level, no topics/embedding_model) → exit 2
    via emit_envelope with code='unsupported_persona_schema'."""
    import io
    import json
    from unittest.mock import patch

    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    v1_data = {
        "name": "oldpersona",
        "role": "analyst",
        "tone": "sharp",
        "structure_pattern": "stitched",
        "revelation_priorities": ["data"],
        "suppression_rules": ["noise"],
        # Deliberately NO topics/embedding_model (v1 shape)
    }
    with open(persona_dir / "oldpersona.yaml", "w") as f:
        yaml.dump(v1_data, f)

    manager = PersonaManager(config_dir=persona_dir)
    buf = io.StringIO()
    with patch("sys.stderr", buf):
        with pytest.raises(SystemExit) as exc_info:
            manager.reload()

    assert exc_info.value.code == 2, f"Expected exit code 2, got {exc_info.value.code}"

    # Verify emit_envelope was called (not a raw logger.error JSON string)
    output = buf.getvalue().strip()
    assert output, "emit_envelope must have written to stderr"
    envelope = json.loads(output)
    assert envelope["code"] == "unsupported_persona_schema"
    assert envelope["level"] == "error"
    assert envelope["stage"] == "persona_load"


def test_invalid_v2_schema_raises_validation_error(tmp_path):
    """A file that is not v1 but has invalid v2 fields raises ValidationError (not swallowed)."""
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    # Missing 'topics' (required field) but has embedding_model → v2 attempt, fails validation
    bad_data = {
        "name": "broken",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        # topics field is missing entirely
    }
    with open(persona_dir / "broken.yaml", "w") as f:
        yaml.dump(bad_data, f)

    manager = PersonaManager(config_dir=persona_dir)
    with pytest.raises(ValidationError):
        manager.reload()


def test_delta_record_status_per_statement():
    """DeltaRecord.status_per_statement accepts Literal values."""
    dr = DeltaRecord(
        baseline_type="claims",
        statements=["A causes B", "C had no effect"],
        status_per_statement=["shared", "suppressed"],
        intent="test",
    )
    assert dr.status_per_statement == ["shared", "suppressed"]


def test_no_try_except_exception_in_reload():
    """Regression: PersonaManager.reload() must NOT contain bare except Exception swallow."""
    import inspect
    import ast
    import textwrap
    src = inspect.getsource(PersonaManager.reload)
    # dedent to remove class-level indentation so ast.parse succeeds
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                raise AssertionError("Found bare 'except:' in PersonaManager.reload")
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                # Check if it just logs and continues (swallows) — we forbid this
                body_stmts = node.body
                is_swallow = all(
                    isinstance(s, (ast.Expr, ast.Continue)) for s in body_stmts
                )
                if is_swallow:
                    raise AssertionError(
                        "PersonaManager.reload() swallows Exception with logger+continue"
                    )
