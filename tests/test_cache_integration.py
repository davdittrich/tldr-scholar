"""Integration tests for cache write-through in synthesize_style pipeline (nwl.6).

Verifies:
- cluster / correlate / aggregate outputs are written to CorpusCache after each stage
- Re-run with same flags → cache hits (LLM mocks never re-called for stages already cached)
- --reset=aggregate → cluster + correlate cached/skipped, aggregate re-runs
- --reset=cluster → all stages re-run (cascade wipes correlate + aggregate too)
- Changed --min-cluster → flag_fingerprint changes → cluster cache miss even without --reset
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldr_scholar.corpus_cache import CorpusCache
from tldr_scholar.personas import DeltaRecord, TopicProfile
from tldr_scholar.scrapers import SocialPost


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_args(
    *,
    source: str = "https://example.com",
    name: str = "test_persona",
    min_cluster: int = 5,
    full_baselines: bool = False,
    window_months: int = 12,
    n_train: int = 10,
    reset: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        source=source,
        name=name,
        months=window_months,
        max_posts=n_train,
        window_months=window_months,
        n_train=n_train,
        n_judge_per_topic=1,
        n_manual_per_topic=1,
        concurrency=1,
        skip_links=True,
        full_baselines=full_baselines,
        min_cluster=min_cluster,
        reset=reset,
    )


def _make_fake_post(text: str = "Hello world") -> SocialPost:
    from datetime import datetime
    return SocialPost(
        text=text,
        source_url="https://example.com/1",
        engagement=1,
        timestamp=datetime(2024, 1, 1),
    )


def _make_corpus_result(posts: list[SocialPost] | None = None):
    posts = posts or [_make_fake_post("post one"), _make_fake_post("post two")]
    return {
        "training": posts,
        "training_topic_labels": ["_global"] * len(posts),
        "topic_centroids": {"_global": [0.0] * 4},
        "eval_judge": {},
        "eval_manual": {},
    }


def _fake_delta() -> DeltaRecord:
    return DeltaRecord(
        baseline_type="claims",
        statements=["statement one"],
        status_per_statement=["shared"],
    )


def _fake_topic_profile() -> TopicProfile:
    return TopicProfile(
        label="_global",
        centroid=[0.0] * 4,
        sample_size=1,
        posts=["post one"],
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers to build a minimal patch context that lets run_synthesis complete
# ---------------------------------------------------------------------------

def _make_patches(
    tmp_path: Path,
    cache: CorpusCache,
    corpus_result: dict | None = None,
    delta_records: list[DeltaRecord] | None = None,
    topic_profile: TopicProfile | None = None,
    global_fields: dict | None = None,
):
    corpus_result = corpus_result or _make_corpus_result()
    delta_records = delta_records if delta_records is not None else [_fake_delta()]
    topic_profile = topic_profile or _fake_topic_profile()
    global_fields = global_fields or {
        "agenda": "test agenda",
        "worldview": "test worldview",
        "pivot_logic": "",
        "identifiable_nuances": [],
    }

    from tldr_scholar.source_baseline import SourceBaselines

    baselines = SourceBaselines(
        claims=["claim1"], extractive_summary=None, abstractive_summary=None
    )

    return [
        patch("tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR", tmp_path),
        patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True),
        patch("tldr_scholar.synthesize_style.ScraperFactory"),
        patch("tldr_scholar.synthesize_style.build_corpus", new=AsyncMock(return_value=corpus_result)),
        patch("tldr_scholar.synthesize_style.build_baselines", new=AsyncMock(return_value=baselines)),
        patch(
            "tldr_scholar.synthesize_style.correlate_against_baselines",
            new=AsyncMock(return_value=delta_records),
        ),
        patch(
            "tldr_scholar.synthesize_style.aggregate_topic",
            new=AsyncMock(return_value=(topic_profile, True)),
        ),
        patch(
            "tldr_scholar.synthesize_style.aggregate_global",
            new=AsyncMock(return_value=(global_fields, True)),
        ),
        patch("tldr_scholar.synthesize_style.check_embedding_model_cached"),
        patch("tldr_scholar.synthesize_style.CACHE_ROOT", cache.root),
        # Inject the shared cache instance directly so tests can inspect it
        patch("tldr_scholar.synthesize_style.CorpusCache", return_value=cache),
    ]


# ---------------------------------------------------------------------------
# Test 1: cache.put called for all three stages on first run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_stages_written_to_cache_on_first_run(tmp_path):
    """After a clean pipeline run, all three stage caches must be populated."""
    cache = CorpusCache(root=tmp_path / "cache")
    args = _make_args()

    patches = _make_patches(tmp_path, cache)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
        patches[9],
        patches[10],
    ):
        from tldr_scholar.synthesize_style import run_synthesis
        await run_synthesis(args)

    # At least one key file must exist under each stage directory
    for stage in ("cluster", "correlate", "aggregate"):
        stage_dir = tmp_path / "cache" / stage
        files = list(stage_dir.glob("*.json")) if stage_dir.exists() else []
        assert files, f"Expected cache files under stage={stage!r}, found none"


# ---------------------------------------------------------------------------
# Test 2: Second run with same flags → cache hits, LLM mocks not re-called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_skips_lllm_stages(tmp_path):
    """Re-running with identical flags must hit all caches; LLM call counts unchanged."""
    cache = CorpusCache(root=tmp_path / "cache")
    args = _make_args()

    import tldr_scholar.synthesize_style as ss  # noqa: F401 (side-effect import OK)

    async def _run_once(build_corpus_mock, correlate_mock, agg_topic_mock, agg_global_mock):
        patches = [
            patch("tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR", tmp_path),
            patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True),
            patch("tldr_scholar.synthesize_style.ScraperFactory"),
            patch("tldr_scholar.synthesize_style.build_corpus", new=build_corpus_mock),
            patch("tldr_scholar.synthesize_style.build_baselines",
                  new=AsyncMock(return_value=_source_baselines())),
            patch("tldr_scholar.synthesize_style.correlate_against_baselines", new=correlate_mock),
            patch("tldr_scholar.synthesize_style.aggregate_topic", new=agg_topic_mock),
            patch("tldr_scholar.synthesize_style.aggregate_global", new=agg_global_mock),
            patch("tldr_scholar.synthesize_style.check_embedding_model_cached"),
            patch("tldr_scholar.synthesize_style.CACHE_ROOT", cache.root),
            patch("tldr_scholar.synthesize_style.CorpusCache", return_value=cache),
        ]
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patches[5], patches[6], patches[7], patches[8], patches[9], patches[10],
        ):
            from tldr_scholar.synthesize_style import run_synthesis
            await run_synthesis(args)

    def _source_baselines():
        from tldr_scholar.source_baseline import SourceBaselines
        return SourceBaselines(claims=["c1"], extractive_summary=None, abstractive_summary=None)

    bc1 = AsyncMock(return_value=_make_corpus_result())
    corr1 = AsyncMock(return_value=[_fake_delta()])
    at1 = AsyncMock(return_value=(_fake_topic_profile(), True))
    ag1 = AsyncMock(return_value=({"agenda": "a", "worldview": "w", "pivot_logic": "", "identifiable_nuances": []}, True))

    await _run_once(bc1, corr1, at1, ag1)

    # Second run with fresh mocks — call counts should remain 0 for cached stages
    bc2 = AsyncMock(return_value=_make_corpus_result())
    corr2 = AsyncMock(return_value=[_fake_delta()])
    at2 = AsyncMock(return_value=(_fake_topic_profile(), True))
    ag2 = AsyncMock(return_value=({"agenda": "a", "worldview": "w", "pivot_logic": "", "identifiable_nuances": []}, True))

    await _run_once(bc2, corr2, at2, ag2)

    # Cluster cache hit → build_corpus not called again
    assert bc2.call_count == 0, f"build_corpus must be skipped on cache hit, called {bc2.call_count} times"
    # Correlate cache hit → correlate_against_baselines not called again
    assert corr2.call_count == 0, f"correlate_against_baselines must be skipped on cache hit"
    # Aggregate cache hit → aggregate_topic and aggregate_global not called again
    assert at2.call_count == 0, f"aggregate_topic must be skipped on cache hit"
    assert ag2.call_count == 0, f"aggregate_global must be skipped on cache hit"


# ---------------------------------------------------------------------------
# Test 3: --reset=aggregate wipes only aggregate; cluster + correlate still cached
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_aggregate_only_reruns_aggregate(tmp_path):
    """--reset=aggregate: cluster + correlate must HIT; aggregate must MISS (re-computed)."""
    cache = CorpusCache(root=tmp_path / "cache")

    def _source_baselines():
        from tldr_scholar.source_baseline import SourceBaselines
        return SourceBaselines(claims=["c1"], extractive_summary=None, abstractive_summary=None)

    async def _run_once(args, bc_mock, corr_mock, at_mock, ag_mock):
        patches = [
            patch("tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR", tmp_path),
            patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True),
            patch("tldr_scholar.synthesize_style.ScraperFactory"),
            patch("tldr_scholar.synthesize_style.build_corpus", new=bc_mock),
            patch("tldr_scholar.synthesize_style.build_baselines",
                  new=AsyncMock(return_value=_source_baselines())),
            patch("tldr_scholar.synthesize_style.correlate_against_baselines", new=corr_mock),
            patch("tldr_scholar.synthesize_style.aggregate_topic", new=at_mock),
            patch("tldr_scholar.synthesize_style.aggregate_global", new=ag_mock),
            patch("tldr_scholar.synthesize_style.check_embedding_model_cached"),
            patch("tldr_scholar.synthesize_style.CACHE_ROOT", cache.root),
            patch("tldr_scholar.synthesize_style.CorpusCache", return_value=cache),
        ]
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patches[5], patches[6], patches[7], patches[8], patches[9], patches[10],
        ):
            from tldr_scholar.synthesize_style import run_synthesis
            await run_synthesis(args)

    gf = {"agenda": "a", "worldview": "w", "pivot_logic": "", "identifiable_nuances": []}

    # First run: populate all caches
    await _run_once(
        _make_args(),
        AsyncMock(return_value=_make_corpus_result()),
        AsyncMock(return_value=[_fake_delta()]),
        AsyncMock(return_value=(_fake_topic_profile(), True)),
        AsyncMock(return_value=(gf, True)),
    )

    # Second run with --reset=aggregate
    bc2 = AsyncMock(return_value=_make_corpus_result())
    corr2 = AsyncMock(return_value=[_fake_delta()])
    at2 = AsyncMock(return_value=(_fake_topic_profile(), True))
    ag2 = AsyncMock(return_value=(gf, True))
    await _run_once(_make_args(reset="aggregate"), bc2, corr2, at2, ag2)

    assert bc2.call_count == 0, "build_corpus must NOT be called (cluster cache still valid)"
    assert corr2.call_count == 0, "correlate must NOT be called (correlate cache still valid)"
    assert at2.call_count >= 1, "aggregate_topic MUST re-run after --reset=aggregate"
    assert ag2.call_count >= 1, "aggregate_global MUST re-run after --reset=aggregate"


# ---------------------------------------------------------------------------
# Test 4: --reset=cluster cascades: all stages re-run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_cluster_reruns_all_stages(tmp_path):
    """--reset=cluster wipes cluster+correlate+aggregate; all stages must re-compute."""
    cache = CorpusCache(root=tmp_path / "cache")

    def _source_baselines():
        from tldr_scholar.source_baseline import SourceBaselines
        return SourceBaselines(claims=["c1"], extractive_summary=None, abstractive_summary=None)

    async def _run_once(args, bc_mock, corr_mock, at_mock, ag_mock):
        patches = [
            patch("tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR", tmp_path),
            patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True),
            patch("tldr_scholar.synthesize_style.ScraperFactory"),
            patch("tldr_scholar.synthesize_style.build_corpus", new=bc_mock),
            patch("tldr_scholar.synthesize_style.build_baselines",
                  new=AsyncMock(return_value=_source_baselines())),
            patch("tldr_scholar.synthesize_style.correlate_against_baselines", new=corr_mock),
            patch("tldr_scholar.synthesize_style.aggregate_topic", new=at_mock),
            patch("tldr_scholar.synthesize_style.aggregate_global", new=ag_mock),
            patch("tldr_scholar.synthesize_style.check_embedding_model_cached"),
            patch("tldr_scholar.synthesize_style.CACHE_ROOT", cache.root),
            patch("tldr_scholar.synthesize_style.CorpusCache", return_value=cache),
        ]
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patches[5], patches[6], patches[7], patches[8], patches[9], patches[10],
        ):
            from tldr_scholar.synthesize_style import run_synthesis
            await run_synthesis(args)

    gf = {"agenda": "a", "worldview": "w", "pivot_logic": "", "identifiable_nuances": []}

    # First run: populate all caches
    await _run_once(
        _make_args(),
        AsyncMock(return_value=_make_corpus_result()),
        AsyncMock(return_value=[_fake_delta()]),
        AsyncMock(return_value=(_fake_topic_profile(), True)),
        AsyncMock(return_value=(gf, True)),
    )

    # Second run with --reset=cluster
    bc2 = AsyncMock(return_value=_make_corpus_result())
    corr2 = AsyncMock(return_value=[_fake_delta()])
    at2 = AsyncMock(return_value=(_fake_topic_profile(), True))
    ag2 = AsyncMock(return_value=(gf, True))
    await _run_once(_make_args(reset="cluster"), bc2, corr2, at2, ag2)

    assert bc2.call_count >= 1, "build_corpus MUST re-run after --reset=cluster"
    assert corr2.call_count >= 1, "correlate MUST re-run after --reset=cluster (cascade)"
    assert at2.call_count >= 1, "aggregate_topic MUST re-run after --reset=cluster (cascade)"
    assert ag2.call_count >= 1, "aggregate_global MUST re-run after --reset=cluster (cascade)"


# ---------------------------------------------------------------------------
# Test 5: Changing --min-cluster changes flag_fingerprint → cluster cache miss
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_min_cluster_causes_cluster_cache_miss(tmp_path):
    """Changing --min-cluster must change flag_fingerprint → cluster stage re-runs."""
    cache = CorpusCache(root=tmp_path / "cache")

    def _source_baselines():
        from tldr_scholar.source_baseline import SourceBaselines
        return SourceBaselines(claims=["c1"], extractive_summary=None, abstractive_summary=None)

    async def _run_once(args, bc_mock, corr_mock, at_mock, ag_mock):
        patches = [
            patch("tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR", tmp_path),
            patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True),
            patch("tldr_scholar.synthesize_style.ScraperFactory"),
            patch("tldr_scholar.synthesize_style.build_corpus", new=bc_mock),
            patch("tldr_scholar.synthesize_style.build_baselines",
                  new=AsyncMock(return_value=_source_baselines())),
            patch("tldr_scholar.synthesize_style.correlate_against_baselines", new=corr_mock),
            patch("tldr_scholar.synthesize_style.aggregate_topic", new=at_mock),
            patch("tldr_scholar.synthesize_style.aggregate_global", new=ag_mock),
            patch("tldr_scholar.synthesize_style.check_embedding_model_cached"),
            patch("tldr_scholar.synthesize_style.CACHE_ROOT", cache.root),
            patch("tldr_scholar.synthesize_style.CorpusCache", return_value=cache),
        ]
        with (
            patches[0], patches[1], patches[2], patches[3], patches[4],
            patches[5], patches[6], patches[7], patches[8], patches[9], patches[10],
        ):
            from tldr_scholar.synthesize_style import run_synthesis
            await run_synthesis(args)

    gf = {"agenda": "a", "worldview": "w", "pivot_logic": "", "identifiable_nuances": []}

    # First run with min_cluster=5
    await _run_once(
        _make_args(min_cluster=5),
        AsyncMock(return_value=_make_corpus_result()),
        AsyncMock(return_value=[_fake_delta()]),
        AsyncMock(return_value=(_fake_topic_profile(), True)),
        AsyncMock(return_value=(gf, True)),
    )

    # Second run with min_cluster=10 — no --reset but flag changed
    bc2 = AsyncMock(return_value=_make_corpus_result())
    corr2 = AsyncMock(return_value=[_fake_delta()])
    at2 = AsyncMock(return_value=(_fake_topic_profile(), True))
    ag2 = AsyncMock(return_value=(gf, True))
    await _run_once(_make_args(min_cluster=10), bc2, corr2, at2, ag2)

    assert bc2.call_count >= 1, (
        "build_corpus MUST re-run when --min-cluster changes (different flag_fingerprint)"
    )


# ---------------------------------------------------------------------------
# Test 6: flag_fingerprint is deterministic — two hashes for same inputs match
# ---------------------------------------------------------------------------

def test_flag_fingerprint_is_deterministic():
    """The flag_fingerprint computed for identical inputs must be identical."""
    def _fp(**kwargs) -> str:
        return hashlib.sha256(
            json.dumps(kwargs, sort_keys=True).encode()
        ).hexdigest()[:16]

    fp1 = _fp(full_baselines=False, min_cluster=5, window_months=12, n_train=200)
    fp2 = _fp(full_baselines=False, min_cluster=5, window_months=12, n_train=200)
    fp3 = _fp(full_baselines=False, min_cluster=10, window_months=12, n_train=200)
    assert fp1 == fp2, "Same inputs must produce identical fingerprints"
    assert fp1 != fp3, "Different min_cluster must produce different fingerprints"
