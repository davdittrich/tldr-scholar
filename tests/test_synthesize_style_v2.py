"""Tests for synthesize_style.py v2-era behaviors — spec test_synthesize_style_v2.py.

Spec coverage row:
  End-to-end with mocked Gemini + small post fixture → persona emitted      [TODO: full e2e]
  Verify JSON error envelope on simulated baseline failure                   [see test_synthesize_style.py]
  Partial-write on simulated exit-4 leaves status=incomplete                [see test_error_contract.py]
  preflight.check_embedding_model_cached invoked at run_synthesis entry     [HERE — gap-closer 2]

Gap-closers from WU-7:
  test_preflight_called_at_run_synthesis_entry — regression guard ensuring
  check_embedding_model_cached() is the first call inside run_synthesis().
"""
from __future__ import annotations

import argparse
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Gap-closer 2: preflight invocation at run_synthesis entry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preflight_called_at_run_synthesis_entry() -> None:
    """check_embedding_model_cached() must be called at the top of run_synthesis().

    Regression guard: if preflight is removed or moved past the ACP_AVAILABLE
    guard it would silently stop running in test environments where ACP_AVAILABLE
    is False.  We verify the call happens before anything else exits.

    Strategy:
    - Patch check_embedding_model_cached as a MagicMock (no-op)
    - Keep ACP_AVAILABLE=False (default in test env) → run_synthesis raises SystemExit(1)
    - Assert the preflight mock was called exactly once before the exit
    """
    from tldr_scholar.synthesize_style import run_synthesis

    args = argparse.Namespace(
        source="https://bsky.app/profile/example.bsky.social",
        name="testpersona",
        window_months=1,
        n_train=10,
        n_judge_per_topic=1,
        n_manual_per_topic=1,
        concurrency=1,
        skip_links=True,
        full_baselines=False,
        reset=None,
        min_cluster=5,
    )

    mock_preflight = MagicMock()

    with (
        patch(
            "tldr_scholar.synthesize_style.check_embedding_model_cached",
            mock_preflight,
        ),
        # Force ACP_AVAILABLE=False so run_synthesis exits at line 109 (after preflight)
        # without needing any scraper/corpus infrastructure.
        patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", False),
    ):
        with pytest.raises(SystemExit) as exc_info:
            await run_synthesis(args)

    assert exc_info.value.code == 1, (
        f"Expected SystemExit(1) from ACP_AVAILABLE=False guard, got {exc_info.value.code}"
    )
    mock_preflight.assert_called_once()
