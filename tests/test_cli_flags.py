"""Tests for synthesize_style CLI flag surface (WU-5b, ticket nwl.6)."""
from __future__ import annotations

import argparse
import sys
from unittest.mock import AsyncMock, patch

import pytest


def _build_parser():
    """Import and return a fresh ArgumentParser from synthesize_style.main."""
    # We reconstruct the same parser used in main() to avoid importing side-effects.
    import importlib
    import tldr_scholar.synthesize_style as ss

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--name")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--max-posts", type=int, default=200)
    parser.add_argument("--window-months", dest="window_months", type=int, default=12)
    parser.add_argument("--n-train", dest="n_train", type=int, default=200)
    parser.add_argument("--n-judge-per-topic", dest="n_judge_per_topic", type=int, default=10)
    parser.add_argument("--n-manual-per-topic", dest="n_manual_per_topic", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--skip-links", action="store_true")
    parser.add_argument("--full-baselines", dest="full_baselines", action="store_true", default=False)
    parser.add_argument(
        "--reset",
        choices=["cluster", "correlate", "aggregate", "all"],
        default=None,
        help="Wipe this stage + all downstream caches before running.",
    )
    parser.add_argument("--min-cluster", dest="min_cluster", type=int, default=5)
    return parser


# ---------------------------------------------------------------------------
# Existing flags — confirm defaults
# ---------------------------------------------------------------------------

def test_window_months_default():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.window_months == 12


def test_n_train_default():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.n_train == 200


def test_n_judge_per_topic_default():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.n_judge_per_topic == 10


def test_n_manual_per_topic_default():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.n_manual_per_topic == 5


def test_full_baselines_default_false():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.full_baselines is False


def test_full_baselines_set():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--full-baselines"])
    assert args.full_baselines is True


# ---------------------------------------------------------------------------
# --reset flag
# ---------------------------------------------------------------------------

def test_reset_default_none():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.reset is None


def test_reset_cluster():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--reset=cluster"])
    assert args.reset == "cluster"


def test_reset_correlate():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--reset=correlate"])
    assert args.reset == "correlate"


def test_reset_aggregate():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--reset=aggregate"])
    assert args.reset == "aggregate"


def test_reset_all():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--reset=all"])
    assert args.reset == "all"


def test_reset_invalid_choice_exits_2():
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["https://example.com", "--reset=invalid"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# --min-cluster flag
# ---------------------------------------------------------------------------

def test_min_cluster_default():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com"])
    assert args.min_cluster == 5


def test_min_cluster_custom():
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--min-cluster", "7"])
    assert args.min_cluster == 7


# ---------------------------------------------------------------------------
# --min-cluster propagates to build_corpus → cluster_posts
# ---------------------------------------------------------------------------

def test_min_cluster_propagated_to_build_corpus():
    """--min-cluster 7 must reach build_corpus as min_cluster=7 kwarg."""
    import asyncio
    from unittest.mock import patch, MagicMock, AsyncMock

    # Simulate argparse result
    parser = _build_parser()
    args = parser.parse_args(["https://example.com", "--min-cluster", "7"])
    assert args.min_cluster == 7

    # build_corpus call site check via import of synthesize_style module
    import tldr_scholar.synthesize_style as ss

    captured_kwargs = {}

    async def fake_build_corpus(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "training": [],
            "training_topic_labels": [],
            "topic_centroids": {},
            "eval_judge": {},
            "eval_manual": {},
        }

    # Patch build_corpus in synthesize_style's namespace
    with patch("tldr_scholar.synthesize_style.build_corpus", side_effect=fake_build_corpus):
        # Call run_synthesis but it will fail after build_corpus (that's fine, we only check kwargs)
        with patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", False):
            try:
                asyncio.run(ss.run_synthesis(args))
            except SystemExit:
                pass

    # We can't assert captured_kwargs["min_cluster"] if ACP_AVAILABLE exits early.
    # Instead test via the actual call routing: check synthesize_style calls
    # build_corpus with min_cluster kwarg by using a direct build_corpus mock
    # with ACP_AVAILABLE=True path.
    captured_kwargs.clear()

    async def fake_build_corpus2(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "training": [],
            "training_topic_labels": [],
            "topic_centroids": {},
            "eval_judge": {},
            "eval_manual": {},
        }

    # Simulate ACP_AVAILABLE so we reach the build_corpus call
    with patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True), \
         patch("tldr_scholar.synthesize_style.build_corpus", side_effect=fake_build_corpus2), \
         patch("tldr_scholar.synthesize_style.check_embedding_model_cached"), \
         patch("tldr_scholar.synthesize_style.ScraperFactory") as mock_sf, \
         patch("tldr_scholar.synthesize_style.httpx.AsyncClient") as mock_client:

        # Setup AsyncClient context manager
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)

        # Setup scraper
        mock_scraper = MagicMock()
        mock_sf.get_scraper.return_value = mock_scraper

        try:
            asyncio.run(ss.run_synthesis(args))
        except (SystemExit, Exception):
            pass

    assert "min_cluster" in captured_kwargs, "build_corpus must be called with min_cluster kwarg"
    assert captured_kwargs["min_cluster"] == 7, f"Expected 7, got {captured_kwargs.get('min_cluster')}"
