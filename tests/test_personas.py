"""Tests for dynamic persona infrastructure."""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path
from tldr_scholar.personas import PersonaManager, Persona

def test_persona_manager_loading(tmp_path):
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    
    # Create a dummy persona
    dav_data = {
        "name": "davdittrich",
        "role": "academic economist",
        "tone": "analytical",
        "structure_pattern": "stitched quotes",
        "hashtag_style": "pascal",
        "revelation_priorities": ["empirical findings"],
        "suppression_rules": ["noise"],
        "attribute_confidence": {"role": 100, "revelation_priorities": 90}
    }
    with open(persona_dir / "davdittrich.yaml", "w") as f:
        yaml.dump(dav_data, f)
        
    manager = PersonaManager(config_dir=persona_dir)
    assert "davdittrich" in manager.list_personas()
    
    p = manager.get_persona("davdittrich")
    assert p.role == "academic economist"
    assert p.hashtag_style == "pascal"
    assert p.revelation_priorities == ["empirical findings"]
    assert p.attribute_confidence["role"] == 100

def test_persona_manager_empty_dir(tmp_path):
    manager = PersonaManager(config_dir=tmp_path / "nonexistent")
    assert manager.list_personas() == []

def test_persona_manager_invalid_yaml(tmp_path):
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    with open(persona_dir / "invalid.yaml", "w") as f:
        f.write("not a: dictionary: structure")
        
    manager = PersonaManager(config_dir=persona_dir)
    assert manager.list_personas() == []
