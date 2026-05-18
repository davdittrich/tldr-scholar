"""Tests for atomic source decomposition.

NOTE: decompose_source was removed in WU-3 code-review (I1 dead-code cleanup).
The replacement is tldr_scholar.source_baseline.build_baselines.
See tests/test_source_baseline.py for current coverage.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="decompose_source removed (I1 dead-code); see test_source_baseline.py"
)
