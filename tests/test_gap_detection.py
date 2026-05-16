"""Tests for domain gap detection in refine_persona."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
from tldr_scholar.refine_persona import detect_profile_gaps

def test_detect_profile_gaps():
    data = {"role": "academic", "agenda": "test"}
    mock_yaml = "- topic_1\n- topic_2"
    
    with patch("tldr_scholar.refine_persona.summarize_via_gemini", return_value=(mock_yaml, None)), \
         patch("tldr_scholar.refine_persona.ACP_AVAILABLE", True):
        gaps = detect_profile_gaps(data)
        
    assert gaps == ["topic_1", "topic_2"]
