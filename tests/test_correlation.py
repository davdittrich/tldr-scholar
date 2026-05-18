"""Tests for atomic post-to-source correlation."""
from __future__ import annotations

import textwrap
from unittest.mock import patch, MagicMock

import pytest

from tldr_scholar.synthesize_style import correlate_post_to_source

@pytest.mark.asyncio
async def test_correlate_post_to_source():
    statements = [
        {"id": "claim_1", "content": "X is a problem", "section": "introduction"},
        {"id": "claim_2", "content": "Y is the cause", "section": "discussion"},
        {"id": "claim_3", "content": "Z is the fix", "section": "conclusion"},
    ]
    post_text = "Check this out: X is a serious issue. Y is definitely why it happens."
    
    mock_response = textwrap.dedent("""
    - statement_id: claim_1
      status: shared
      intent: Highlight the problem severity.
    - statement_id: claim_2
      status: shared
      intent: Confirm the causal mechanism.
    - statement_id: claim_3
      status: suppressed
      intent: User usually avoids speculative fixes.
    """).strip()
    
    with patch("tldr_scholar.synthesize_style.summarize_via_gemini", return_value=(mock_response, None)), \
         patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True):
        deltas = await correlate_post_to_source(statements, post_text)
        
    assert len(deltas) == 3
    assert deltas[0]["statement_id"] == "claim_1"
    assert deltas[0]["status"] == "shared"
    assert deltas[2]["status"] == "suppressed"
