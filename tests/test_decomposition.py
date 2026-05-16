"""Tests for atomic source decomposition."""
from __future__ import annotations

import textwrap
from unittest.mock import patch, MagicMock
from tldr_scholar.synthesize_style import decompose_source

def test_decompose_source_extracts_claims():
    text = "Introduction: X is a problem. Discussion: Y is the cause. Conclusion: Z is the fix."
    
    mock_response = textwrap.dedent("""
    - id: claim_1
      content: X is a problem
      section: introduction
    - id: claim_2
      content: Y is the cause
      section: discussion
    - id: claim_3
      content: Z is the fix
      section: conclusion
    """).strip()
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_response, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True):
        claims = decompose_source(text)
        
    assert len(claims) == 3
    assert claims[0]["id"] == "claim_1"
    assert claims[0]["section"] == "introduction"
    assert "X is a problem" in claims[0]["content"]
