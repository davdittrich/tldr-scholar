"""Tests for synthesize_style JSONL pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import directly from file
from tldr_scholar.synthesize_style import main

def test_jsonl_pipeline(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"post": "Post content", "url": "http://example.com"}) + "\n")
    
    # Mock return values for the pipeline
    mock_decomp = [{"id": "c1", "content": "Claim 1", "section": "intro"}]
    mock_corr = [{"statement_id": "c1", "status": "shared", "intent": "Test"}]
    mock_synth = {
        "profile": {"agenda": "Test agenda", "role": "tester", "tone": "t", "structure_pattern": "stitched"},
        "confidence": {"agenda": 100}
    }
    
    with patch("tldr_scholar.synthesize_style.ingest", return_value=("Source text", "text")), \
         patch("tldr_scholar.synthesize_style.decompose_source", return_value=mock_decomp), \
         patch("tldr_scholar.synthesize_style.correlate_post_to_source", return_value=mock_corr), \
         patch("tldr_scholar.synthesize_style.synthesize_deep_profile", return_value=mock_synth), \
         patch("tldr_scholar.synthesize_style.summarize_via_gemini"), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True), \
         patch("sys.argv", ["synthesize_style.py", str(corpus), "--format", "jsonl", "--name", "test-deep"]), \
         patch("tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR", tmp_path):
        main()
        
    output_file = tmp_path / "test-deep.yaml"
    assert output_file.exists()
