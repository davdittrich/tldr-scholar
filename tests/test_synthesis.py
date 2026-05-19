"""Tests for aggregate synthesis from atomic deltas."""
from __future__ import annotations

import pytest

# synthesize_deep_profile removed in tldr-scholar-bbi (dead code since WU-4).
# Rewrite owned by tldr-scholar-gi5.

# decompose_source and correlate_post_to_source removed (I1 dead-code cleanup).
# Replacements: build_baselines (test_source_baseline.py), correlate_against_baselines (test_correlator.py).

@pytest.mark.asyncio
@pytest.mark.skip(reason="decompose_source removed (I1 dead-code); see test_source_baseline.py")
async def test_decompose_source_extracts_claims():
    pass


@pytest.mark.asyncio
@pytest.mark.skip(reason="correlate_post_to_source removed (I1 dead-code); see test_correlator.py")
async def test_correlate_post_to_source_finds_deltas():
    pass


@pytest.mark.asyncio
@pytest.mark.skip(reason="dead-code path removed in tldr-scholar-bbi; rewrite owned by tldr-scholar-gi5")
async def test_synthesize_deep_profile():
    pass
