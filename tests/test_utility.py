"""Tests for style synthesis utility."""
from __future__ import annotations

import yaml
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

from tldr_scholar import synthesize_style as ss

def test_synthesize_style_logic(tmp_path):
    source = tmp_path / "samples.txt"
    source.write_text("Some text samples.")
    
    output_file = tmp_path / "persona.yaml"
    
    # Mocks for atomic pipeline
    mock_decomp = "- id: c1\n  claim: test"
    mock_corr = "- statement_id: c1\n  status: shared"
    mock_synth = textwrap.dedent("""
    profile:
      name: test-persona
      role: tester
      tone: robotic
      structure_pattern: fixed
    confidence:
      tone: 100
    """).strip()
    
    mock_gemini = MagicMock(side_effect=[
        (mock_decomp, None), # decompose_source
        (mock_corr, None),   # correlate_post_to_source
        (mock_synth, None)   # synthesize_deep_profile
    ])
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", mock_gemini), \
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
    
    mock_gemini = MagicMock(side_effect=[
        ("- id: c1\n  claim: test", None),
        ("- statement_id: c1\n  status: shared", None),
        ("just a string", None)
    ])
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", mock_gemini), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True), \
         patch("sys.argv", ["synthesize-style.py", str(source)]), \
         patch("sys.exit") as mock_exit:
        ss.main()
        mock_exit.assert_called_with(1)
