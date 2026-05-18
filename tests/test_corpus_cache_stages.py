"""Tests for per-stage CorpusCache with cascade invalidation (WU-5b, ticket nwl.6)."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from tldr_scholar.corpus_cache import CorpusCache, STAGES, STAGE_CASCADE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key(url: str = "https://example.com", embedding_model: str = "test-model") -> dict:
    return {
        "url": url,
        "embedding_model": embedding_model,
        "flag_fingerprint": "abc123",
        "corpus_hash": "deadzero",
    }


# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------

def test_stages_tuple_order():
    assert STAGES == ("scrape", "cluster", "correlate", "aggregate")


def test_stage_cascade_cluster_includes_downstream():
    assert STAGE_CASCADE["cluster"] == ["cluster", "correlate", "aggregate"]


def test_stage_cascade_correlate_includes_downstream():
    assert STAGE_CASCADE["correlate"] == ["correlate", "aggregate"]


def test_stage_cascade_aggregate_only():
    assert STAGE_CASCADE["aggregate"] == ["aggregate"]


def test_stage_cascade_scrape_includes_all():
    assert STAGE_CASCADE["scrape"] == ["scrape", "cluster", "correlate", "aggregate"]


def test_stage_cascade_all_is_all_stages():
    assert set(STAGE_CASCADE["all"]) == set(STAGES)


# ---------------------------------------------------------------------------
# Cache key includes embedding_model and flag_fingerprint
# ---------------------------------------------------------------------------

def test_cache_key_includes_embedding_model():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key1 = _make_key(embedding_model="model-a")
        key2 = _make_key(embedding_model="model-b")
        h1 = cache._key_hash("cluster", key1)
        h2 = cache._key_hash("cluster", key2)
        assert h1 != h2, "Different embedding_model must produce different cache keys"


def test_cache_key_includes_flag_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key1 = {**_make_key(), "flag_fingerprint": "fp1"}
        key2 = {**_make_key(), "flag_fingerprint": "fp2"}
        h1 = cache._key_hash("aggregate", key1)
        h2 = cache._key_hash("aggregate", key2)
        assert h1 != h2, "Different flag_fingerprint must produce different cache keys"


def test_cache_key_includes_stage():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        h1 = cache._key_hash("cluster", key)
        h2 = cache._key_hash("correlate", key)
        assert h1 != h2, "Same key_dict but different stage must produce different hashes"


# ---------------------------------------------------------------------------
# put / get round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage", STAGES)
def test_put_get_roundtrip(stage):
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        value = {"data": [1, 2, 3], "stage": stage}
        cache.put(stage, key, value)
        result = cache.get(stage, key)
        assert result == value


def test_get_returns_none_on_miss():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        result = cache.get("cluster", _make_key())
        assert result is None


def test_put_overwrites_existing():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        cache.put("aggregate", key, {"v": 1})
        cache.put("aggregate", key, {"v": 2})
        assert cache.get("aggregate", key) == {"v": 2}


# ---------------------------------------------------------------------------
# invalidate cascade semantics
# ---------------------------------------------------------------------------

def _populate_all_stages(cache: CorpusCache, key: dict) -> None:
    for stage in STAGES:
        cache.put(stage, key, {"stage": stage})


def test_invalidate_cluster_removes_cluster_correlate_aggregate():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        _populate_all_stages(cache, key)

        cache.invalidate("cluster")

        assert cache.get("scrape", key) is not None, "scrape must be preserved"
        assert cache.get("cluster", key) is None, "cluster must be wiped"
        assert cache.get("correlate", key) is None, "correlate must be wiped"
        assert cache.get("aggregate", key) is None, "aggregate must be wiped"


def test_invalidate_correlate_removes_correlate_aggregate_only():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        _populate_all_stages(cache, key)

        cache.invalidate("correlate")

        assert cache.get("scrape", key) is not None, "scrape must be preserved"
        assert cache.get("cluster", key) is not None, "cluster must be preserved"
        assert cache.get("correlate", key) is None, "correlate must be wiped"
        assert cache.get("aggregate", key) is None, "aggregate must be wiped"


def test_invalidate_aggregate_removes_aggregate_only():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        _populate_all_stages(cache, key)

        cache.invalidate("aggregate")

        assert cache.get("scrape", key) is not None, "scrape must be preserved"
        assert cache.get("cluster", key) is not None, "cluster must be preserved"
        assert cache.get("correlate", key) is not None, "correlate must be preserved"
        assert cache.get("aggregate", key) is None, "aggregate must be wiped"


def test_invalidate_all_removes_everything():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        _populate_all_stages(cache, key)

        cache.invalidate("all")

        for stage in STAGES:
            assert cache.get(stage, key) is None, f"{stage} must be wiped"


def test_invalidate_scrape_removes_all():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        _populate_all_stages(cache, key)

        cache.invalidate("scrape")

        for stage in STAGES:
            assert cache.get(stage, key) is None, f"{stage} must be wiped by scrape cascade"


def test_invalidate_idempotent_on_missing_dirs():
    """invalidate must not raise if stage dirs don't exist yet."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        # No puts — dirs do not exist
        cache.invalidate("all")  # must not raise


# ---------------------------------------------------------------------------
# Backwards-compat scrape shim (old get(url, months) / put(url, months, posts))
# ---------------------------------------------------------------------------

def test_old_scrape_api_get_miss():
    """Legacy get(url, months) returns None on miss (shim preserved)."""
    from tldr_scholar.scrapers import SocialPost
    from datetime import datetime, timezone

    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        result = cache.get("https://example.com", 12)
        assert result is None


def test_old_scrape_api_put_then_get():
    """Legacy put(url, months, posts) / get(url, months) round-trip."""
    from tldr_scholar.scrapers import SocialPost
    from datetime import datetime, timezone

    post = SocialPost(
        text="hello",
        timestamp=datetime.now(timezone.utc),
        source_url="https://example.com/post/1",
        links=[],
        engagement=0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        cache.put("https://example.com", 12, [post])
        result = cache.get("https://example.com", 12)
        assert result is not None
        assert len(result) == 1
        assert result[0].text == "hello"


# ---------------------------------------------------------------------------
# Resume-from-incomplete simulation
# ---------------------------------------------------------------------------

def test_resume_from_incomplete_resets_aggregate_only():
    """Simulate exit-4: incomplete persona written with status=incomplete.
    Re-run with --reset=aggregate: aggregate cache wiped, cluster+correlate preserved.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()

        # Simulate completed cluster + correlate stages from first run
        cache.put("cluster", key, {"clusters": ["topic_0", "topic_1"]})
        cache.put("correlate", key, {"deltas": [{"baseline_type": "claims"}]})
        # aggregate was NOT written (simulates incomplete run)

        # Verify state
        assert cache.get("cluster", key) is not None
        assert cache.get("correlate", key) is not None
        assert cache.get("aggregate", key) is None

        # --reset=aggregate: drop only aggregate
        cache.invalidate("aggregate")

        # cluster + correlate still present; aggregate still None (was never written)
        assert cache.get("cluster", key) is not None, "cluster cache preserved after --reset=aggregate"
        assert cache.get("correlate", key) is not None, "correlate cache preserved after --reset=aggregate"
        assert cache.get("aggregate", key) is None, "aggregate cache absent (never written)"


def test_resume_from_incomplete_with_stale_aggregate_cleared():
    """Simulate partial aggregate written then re-run with --reset=aggregate."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()

        cache.put("cluster", key, {"clusters": ["topic_0"]})
        cache.put("correlate", key, {"deltas": []})
        cache.put("aggregate", key, {"topics": {}})  # stale partial

        cache.invalidate("aggregate")

        assert cache.get("cluster", key) is not None, "cluster preserved"
        assert cache.get("correlate", key) is not None, "correlate preserved"
        assert cache.get("aggregate", key) is None, "stale aggregate wiped"


def test_stage_get_handles_corrupt_cache_file():
    """Corrupt stage cache file → treated as miss, file deleted."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(root=Path(tmp))
        key = _make_key()
        cache.put("cluster", key, {"data": "ok"})

        # Find and corrupt the cache file
        cache_files = list((Path(tmp) / "cluster").glob("*.json"))
        assert len(cache_files) == 1
        cache_files[0].write_text("{not valid json")

        # Read should return None, not raise
        assert cache.get("cluster", key) is None
        # Corrupt file should be cleaned up
        assert not cache_files[0].exists()
