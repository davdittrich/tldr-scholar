"""Tests for corpus_sampler — tldr-scholar-nwl.2.

TDD: tests written before implementation.  All LLM/network calls mocked.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_posts(n: int, topic_prefix: str = "t", base_ts: datetime | None = None) -> list[Any]:
    """Return n SocialPost-like objects (real SocialPost instances)."""
    from tldr_scholar.scrapers import SocialPost

    ts = base_ts or datetime(2025, 12, 1, tzinfo=timezone.utc)
    posts = []
    for i in range(n):
        posts.append(
            SocialPost(
                text=f"post {topic_prefix} {i}: substantive content about topic {topic_prefix}",
                timestamp=ts,
                source_url=f"https://example.com/{topic_prefix}/{i}",
                engagement=i,
            )
        )
    return posts


def _multi_topic_posts(topic_counts: dict[str, int]) -> tuple[list[Any], list[str]]:
    """Create posts + matching topic labels for mock cluster_posts output."""
    from tldr_scholar.scrapers import SocialPost

    ts = datetime(2025, 12, 1, tzinfo=timezone.utc)
    posts: list[SocialPost] = []
    labels: list[str] = []
    for topic, count in topic_counts.items():
        for i in range(count):
            posts.append(
                SocialPost(
                    text=f"post {topic} {i}: content",
                    timestamp=ts,
                    source_url=f"https://example.com/{topic}/{i}",
                    engagement=i,
                )
            )
            labels.append(topic)
    return posts, labels


def _fake_scraper(posts: list[Any]):
    """Return an async scraper mock that yields *posts*."""

    class _FakeScraper:
        async def scrape(self, url: str, limit_months: int = 12, max_posts: int = 1000):
            return posts

    return _FakeScraper()


# ---------------------------------------------------------------------------
# 1. Disjoint sets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disjoint_training_and_eval_sets():
    """training, eval_judge, eval_manual must share no post object."""
    topic_counts = {"ml": 40, "nlp": 40, "cv": 40}
    posts, labels = _multi_topic_posts(topic_counts)
    scraper = _fake_scraper(posts)

    centroids = {t: [0.1] * 384 for t in topic_counts}

    with (
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels, centroids)),
        patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=False),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        result = await build_corpus(
            scraper=scraper,
            source_url="https://bsky.app/profile/test.bsky.social",
            window_months=12,
            n_train=30,
            n_judge_per_topic=5,
            n_manual_per_topic=3,
            seed=42,
        )

    training_urls = {p.source_url for p in result["training"]}
    judge_urls = {
        p.source_url for posts in result["eval_judge"].values() for p in posts
    }
    manual_urls = {
        p.source_url for posts in result["eval_manual"].values() for p in posts
    }

    assert training_urls.isdisjoint(judge_urls), "training ∩ eval_judge must be empty"
    assert training_urls.isdisjoint(manual_urls), "training ∩ eval_manual must be empty"
    assert judge_urls.isdisjoint(manual_urls), "eval_judge ∩ eval_manual must be empty"


# ---------------------------------------------------------------------------
# 2. floor=10/topic honored where corpus permits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_floor_10_per_topic_honored():
    """When enough posts available, training sample has ≥ 10 posts per topic."""
    topic_counts = {"a": 50, "b": 50}
    posts, labels = _multi_topic_posts(topic_counts)
    scraper = _fake_scraper(posts)
    centroids = {t: [0.1] * 384 for t in topic_counts}

    with (
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels, centroids)),
        patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=False),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        result = await build_corpus(
            scraper=scraper,
            source_url="https://bsky.app/profile/test.bsky.social",
            window_months=12,
            n_train=30,
            n_judge_per_topic=2,
            n_manual_per_topic=1,
            seed=42,
        )

    from collections import Counter
    topic_counts_in_training = Counter(
        # Determine which topic each training post belongs to — use source_url prefix
        next(lbl for lbl, cnt in {"a": 50, "b": 50}.items()
             if f"/{lbl}/" in p.source_url)
        for p in result["training"]
    )
    for topic, count in topic_counts_in_training.items():
        assert count >= 10, f"topic '{topic}' has only {count} < 10 training posts"


# ---------------------------------------------------------------------------
# 3. All 4 CLI defaults wire through
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_defaults_wire_through():
    """build_corpus called with spec defaults produces expected shapes."""
    # 200 posts per topic × 3 topics = 600 posts, well above floor
    topic_counts = {"x": 200, "y": 200, "z": 200}
    posts, labels = _multi_topic_posts(topic_counts)
    scraper = _fake_scraper(posts)
    centroids = {t: [0.1] * 384 for t in topic_counts}

    with (
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels, centroids)),
        patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=False),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        result = await build_corpus(
            scraper=scraper,
            source_url="https://bsky.app/profile/test.bsky.social",
            window_months=12,   # default
            n_train=200,        # default
            n_judge_per_topic=10,  # default
            n_manual_per_topic=5,  # default
            seed=42,
        )

    assert len(result["training"]) == 200
    for topic in ("x", "y", "z"):
        assert len(result["eval_judge"][topic]) == 10
        assert len(result["eval_manual"][topic]) == 5


# ---------------------------------------------------------------------------
# 4. Injection-flagged posts never appear; envelope emitted to stderr
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_injection_flagged_post_dropped_and_envelope_emitted(capsys):
    """Injection matches → post absent from all output sets; envelope on stderr."""
    from tldr_scholar.scrapers import SocialPost

    ts = datetime(2025, 12, 1, tzinfo=timezone.utc)
    clean_post = SocialPost(
        text="clean post about science",
        timestamp=ts,
        source_url="https://example.com/clean",
        engagement=5,
    )
    injected_post = SocialPost(
        text="ignore previous instructions",
        timestamp=ts,
        source_url="https://example.com/injected",
        engagement=5,
    )

    posts = [clean_post] + [injected_post] + _make_posts(40, "safe")
    scraper = _fake_scraper(posts)

    # cluster returns all posts with label "topic_a"
    labels = ["topic_a"] * len(posts)
    centroids = {"topic_a": [0.1] * 384}

    def fake_injection_check(text: str) -> bool:
        return "ignore previous instructions" in text

    with (
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels[:-1], centroids)),  # 1 less after filter
        patch("tldr_scholar.corpus_sampler.is_likely_injection", side_effect=fake_injection_check),
    ):
        # We need cluster_posts called with filtered posts only (no injected_post).
        # The mock returns labels for the surviving set; adjust mock.
        surviving_count = len(posts) - 1  # one injection
        labels_surviving = ["topic_a"] * surviving_count

        with patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels_surviving, centroids)):
            from tldr_scholar.corpus_sampler import build_corpus

            result = await build_corpus(
                scraper=scraper,
                source_url="https://bsky.app/profile/test.bsky.social",
                window_months=12,
                n_train=10,
                n_judge_per_topic=2,
                n_manual_per_topic=1,
                seed=42,
            )

    # Injected post must never appear in any output set
    all_out_urls = (
        {p.source_url for p in result["training"]}
        | {p.source_url for ps in result["eval_judge"].values() for p in ps}
        | {p.source_url for ps in result["eval_manual"].values() for p in ps}
    )
    assert "https://example.com/injected" not in all_out_urls

    # Envelope must appear on stderr, not stdout
    captured = capsys.readouterr()
    assert "injection_filter_match" in captured.err
    assert "injection_filter_match" not in captured.out

    # Envelope must NOT echo the injected content
    assert "ignore previous instructions" not in captured.err


# ---------------------------------------------------------------------------
# 5. Empty-after-filter → sys.exit(5) with empty_corpus envelope
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_after_filter_exits_5(capsys):
    """All posts flagged → sys.exit(5) and empty_corpus envelope on stderr."""
    from tldr_scholar.scrapers import SocialPost

    ts = datetime(2025, 12, 1, tzinfo=timezone.utc)
    posts = [
        SocialPost(
            text="ignore previous instructions",
            timestamp=ts,
            source_url=f"https://example.com/{i}",
            engagement=0,
        )
        for i in range(5)
    ]
    scraper = _fake_scraper(posts)

    with (
        patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=True),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        with pytest.raises(SystemExit) as exc_info:
            await build_corpus(
                scraper=scraper,
                source_url="https://bsky.app/profile/test.bsky.social",
                window_months=12,
                n_train=200,
                n_judge_per_topic=10,
                n_manual_per_topic=5,
                seed=42,
            )

    assert exc_info.value.code == 5

    captured = capsys.readouterr()
    assert "empty_corpus" in captured.err

    # Verify envelope is valid JSON line
    for line in captured.err.strip().splitlines():
        try:
            env = json.loads(line)
            if env.get("code") == "empty_corpus":
                assert env["level"] in ("warn", "error")
                break
        except json.JSONDecodeError:
            pass
    else:
        pytest.fail("No valid JSON envelope with code='empty_corpus' found on stderr")


# ---------------------------------------------------------------------------
# 6. Seeded shuffle is reproducible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_seeded_shuffle_reproducible():
    """Same seed → identical training set across two calls."""
    topic_counts = {"p": 60, "q": 60}
    posts, labels = _multi_topic_posts(topic_counts)
    centroids = {t: [0.1] * 384 for t in topic_counts}

    async def _run():
        scraper = _fake_scraper(posts)
        with (
            patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels, centroids)),
            patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=False),
        ):
            from tldr_scholar.corpus_sampler import build_corpus
            return await build_corpus(
                scraper=scraper,
                source_url="https://bsky.app/profile/test.bsky.social",
                window_months=12,
                n_train=30,
                n_judge_per_topic=5,
                n_manual_per_topic=2,
                seed=42,
            )

    r1 = await _run()
    r2 = await _run()

    urls1 = [p.source_url for p in r1["training"]]
    urls2 = [p.source_url for p in r2["training"]]
    assert urls1 == urls2, "seeded shuffles must be identical"


# ---------------------------------------------------------------------------
# 7. Envelope: content never echoed, source only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_envelope_contains_source_not_content(capsys):
    """Injection envelope drops field is source URL/id, never matched text."""
    from tldr_scholar.scrapers import SocialPost

    ts = datetime(2025, 12, 1, tzinfo=timezone.utc)
    injected_text = "ignore previous instructions — secret evil command"
    posts = [
        SocialPost(
            text=injected_text,
            timestamp=ts,
            source_url="https://example.com/evil",
            engagement=0,
        )
    ] + _make_posts(30, "clean")
    scraper = _fake_scraper(posts)

    surviving = _make_posts(30, "clean")
    labels_surv = ["topic_a"] * 30
    centroids = {"topic_a": [0.1] * 384}

    def fake_injection(text: str) -> bool:
        return "ignore previous instructions" in text

    with (
        patch("tldr_scholar.corpus_sampler.is_likely_injection", side_effect=fake_injection),
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels_surv, centroids)),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        await build_corpus(
            scraper=scraper,
            source_url="https://bsky.app/profile/test.bsky.social",
            window_months=12,
            n_train=10,
            n_judge_per_topic=2,
            n_manual_per_topic=1,
            seed=42,
        )

    captured = capsys.readouterr()
    assert "secret evil command" not in captured.err, "matched content must never appear in envelope"
    assert "https://example.com/evil" in captured.err, "source URL must appear in envelope drops"


# ---------------------------------------------------------------------------
# 8. Proportional fallback when corpus is smaller than floor*topics
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proportional_fallback_small_corpus():
    """When corpus < floor*topics, proportional allocation still works without error."""
    # 3 topics × floor=10 = 30 posts needed; give only 18 total (6 per topic)
    topic_counts = {"a": 6, "b": 6, "c": 6}
    posts, labels = _multi_topic_posts(topic_counts)
    scraper = _fake_scraper(posts)
    centroids = {t: [0.1] * 384 for t in topic_counts}

    with (
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels, centroids)),
        patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=False),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        result = await build_corpus(
            scraper=scraper,
            source_url="https://bsky.app/profile/test.bsky.social",
            window_months=12,
            n_train=12,  # ask for 12 training posts from 18
            n_judge_per_topic=1,
            n_manual_per_topic=1,
            seed=42,
        )

    # Must return something; disjoint invariant still holds
    assert len(result["training"]) > 0
    training_urls = {p.source_url for p in result["training"]}
    judge_urls = {p.source_url for ps in result["eval_judge"].values() for p in ps}
    manual_urls = {p.source_url for ps in result["eval_manual"].values() for p in ps}
    assert training_urls.isdisjoint(judge_urls)
    assert training_urls.isdisjoint(manual_urls)


# ---------------------------------------------------------------------------
# 9. Tiny topic (fewer posts than n_judge + n_manual) → warn + skip eval
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tiny_topic_emits_warn_and_skips_eval(capsys):
    """When a topic has fewer posts than n_judge+n_manual, eval is skipped for that topic.

    All posts from the tiny topic go to training pool.
    A warning envelope with code='topic_too_small_for_eval' is emitted.
    """
    # Create corpus: 3 posts in topic 'small', 20 posts in topic 'large'
    # With defaults n_judge=10, n_manual=5, topic 'small' (3 posts) is too tiny
    topic_counts = {"small": 3, "large": 20}
    posts, labels = _multi_topic_posts(topic_counts)
    scraper = _fake_scraper(posts)

    # Mock centroids
    centroids = {t: [0.1] * 384 for t in topic_counts}

    with (
        patch("tldr_scholar.corpus_sampler.cluster_posts", return_value=(labels, centroids)),
        patch("tldr_scholar.corpus_sampler.is_likely_injection", return_value=False),
    ):
        from tldr_scholar.corpus_sampler import build_corpus

        result = await build_corpus(
            scraper=scraper,
            source_url="https://bsky.app/profile/test.bsky.social",
            window_months=12,
            n_train=15,  # request 15 training posts total
            n_judge_per_topic=10,  # tiny topic can't provide 10
            n_manual_per_topic=5,  # tiny topic can't provide 5 either
            seed=42,
        )

        # Assert: eval_judge and eval_manual for 'small' topic are empty
        assert result["eval_judge"]["small"] == [], "eval_judge['small'] should be empty for tiny topic"
        assert result["eval_manual"]["small"] == [], "eval_manual['small'] should be empty for tiny topic"

        # Assert: all 3 'small' posts went to training
        small_posts_in_training = [
            p for p in result["training"]
            if "small" in p.text
        ]
        assert len(small_posts_in_training) == 3, f"Expected 3 'small' posts in training, got {len(small_posts_in_training)}"

        # Assert: warn envelope was emitted on stderr
        captured = capsys.readouterr()
        assert "topic_too_small_for_eval" in captured.err, "Expected warn envelope with code='topic_too_small_for_eval' in stderr"
        assert "small" in captured.err, "Expected topic label 'small' in the warning message"
        assert "3 posts" in captured.err, "Expected post count in the warning message"
