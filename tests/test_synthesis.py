"""Tests for aggregate synthesis from atomic deltas."""
from __future__ import annotations

import textwrap
from unittest.mock import patch, MagicMock
from tldr_scholar.synthesize_style import synthesize_deep_profile

def test_synthesize_deep_profile():
    correlation_reports = [
        [
            {"statement_id": "c1", "status": "shared", "intent": "Important data"},
            {"statement_id": "c2", "status": "suppressed", "intent": "Noise"},
        ],
        [
            {"statement_id": "c3", "status": "shared", "intent": "Data point"},
            {"statement_id": "c4", "status": "suppressed", "intent": "Noise"},
        ]
    ]
    
    mock_yaml = textwrap.dedent("""
    profile:
      agenda: Extract data
      worldview: Empirical
      revelation_priorities: [empirical findings]
      suppression_rules: [noise, future work]
    confidence:
      agenda: 90
      worldview: 85
      revelation_priorities: 95
      suppression_rules: 100
    """).strip()
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_yaml, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True):
        result = synthesize_deep_profile(correlation_reports)
        
    assert "profile" in result
    assert "confidence" in result
    assert result["confidence"]["suppression_rules"] == 100
