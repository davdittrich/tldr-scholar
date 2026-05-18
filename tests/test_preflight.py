"""Tests for preflight model cache check."""
from __future__ import annotations

import sys
import types
from unittest.mock import patch


def test_check_embedding_model_cached_warns_when_not_cached():
    """When model not on disk, emits a loguru warning (does NOT raise or exit)."""
    from tldr_scholar.preflight import check_embedding_model_cached

    warnings_emitted: list[str] = []

    import loguru
    sink_id = loguru.logger.add(
        lambda msg: warnings_emitted.append(msg),
        level="WARNING",
        format="{message}",
    )
    try:
        with patch("tldr_scholar.preflight._model_is_cached", return_value=False):
            check_embedding_model_cached()  # must not raise
    finally:
        loguru.logger.remove(sink_id)

    assert len(warnings_emitted) > 0, "Expected at least one loguru WARNING"
    combined = " ".join(warnings_emitted).lower()
    assert "embedding" in combined or "model" in combined


def test_check_embedding_model_cached_no_warning_when_cached():
    """When model IS on disk, no warning is emitted."""
    from tldr_scholar.preflight import check_embedding_model_cached

    warnings_emitted: list[str] = []

    import loguru
    sink_id = loguru.logger.add(
        lambda msg: warnings_emitted.append(msg),
        level="WARNING",
        format="{message}",
    )
    try:
        with patch("tldr_scholar.preflight._model_is_cached", return_value=True):
            check_embedding_model_cached()
    finally:
        loguru.logger.remove(sink_id)

    assert len(warnings_emitted) == 0, f"Unexpected warnings: {warnings_emitted}"


def test_check_embedding_model_cached_is_callable():
    """preflight.check_embedding_model_cached is importable and callable."""
    from tldr_scholar import preflight
    assert callable(preflight.check_embedding_model_cached)
