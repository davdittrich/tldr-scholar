"""Tests for style synthesis utility."""
from __future__ import annotations

import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

from tldr_scholar import synthesize_style as ss

def test_synthesize_style_logic(tmp_path):
    source = tmp_path / "samples.txt"
    source.write_text("Some text samples.")
    
    output_file = tmp_path / "persona.yaml"
    
    mock_yaml = """
name: test-persona
role: tester
tone: robotic
structure_pattern: fixed
hashtag_style: lowercase
"""
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_yaml, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True), \
         patch("sys.argv", ["synthesize-style.py", str(source), "--output", str(output_file)]):
        ss.main()
    
    assert output_file.exists()
    data = yaml.safe_load(output_file.read_text())
    assert data["name"] == "test-persona"
    assert data["role"] == "tester"

def test_yaml_cleaning():
    pass

def test_isinstance_check(tmp_path):
    source = tmp_path / "samples.txt"
    source.write_text("Some text samples.")
    
    # Non-dict output
    mock_yaml = "just a string"
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_yaml, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True), \
         patch("sys.argv", ["synthesize-style.py", str(source)]), \
         patch("sys.exit") as mock_exit:
        ss.main()
        mock_exit.assert_called_with(1)
