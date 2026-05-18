"""Tests for atomic post-to-source correlation.

NOTE: correlate_post_to_source was removed in WU-3 code-review (I1 dead-code
cleanup). The replacement is tldr_scholar.correlator.correlate_against_baselines.
See tests/test_correlator.py for current coverage.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="correlate_post_to_source removed (I1 dead-code); see test_correlator.py"
)
