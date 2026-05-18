"""Tests for aggregate synthesis from atomic deltas."""
from __future__ import annotations

import textwrap
from unittest.mock import patch, MagicMock

import pytest

from tldr_scholar.synthesize_style import (
    synthesize_deep_profile,
    decompose_source,
    correlate_post_to_source
)

@pytest.mark.asyncio
async def test_decompose_source_extracts_claims():
    text = "The study found that A causes B. However, C had no effect."
    mock_yaml = textwrap.dedent("""
    - id: c1
      claim: A causes B
      type: finding
    - id: c2
      claim: C had no effect
      type: null_result
    """).strip()

    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_yaml, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True):
        result = await decompose_source(text)

    assert len(result) == 2
    assert result[0]["id"] == "c1"
    assert "A causes B" in result[0]["claim"]

@pytest.mark.asyncio
async def test_correlate_post_to_source_finds_deltas():
    statements = [{"id": "c1", "claim": "A causes B"}]
    post_text = "I saw that A leads to B!"
    mock_yaml = textwrap.dedent("""
    - statement_id: c1
      status: shared
      intent: Direct revelation
    """).strip()

    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_yaml, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True):
        result = await correlate_post_to_source(statements, post_text)

    assert len(result) == 1
    assert result[0]["status"] == "shared"

@pytest.mark.asyncio
async def test_synthesize_deep_profile():
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
        result = await synthesize_deep_profile(correlation_reports)
        
    assert "profile" in result
    assert "confidence" in result
    assert result["confidence"]["suppression_rules"] == 100
