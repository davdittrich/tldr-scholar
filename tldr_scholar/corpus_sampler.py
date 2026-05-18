"""Corpus sampling pipeline for tldr-scholar training + eval splits.

Scrape → injection-filter → cluster → topic-balanced sampler with
per-topic eval holdouts.

Exit codes:
    5 — training set empty after scrape + injection filter
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from tldr_scholar.error_contract import emit_envelope as emit
from tldr_scholar.scrape_filter import is_likely_injection
from tldr_scholar.topic_cluster import cluster_posts

if TYPE_CHECKING:
    from tldr_scholar.scrapers import SocialPost

# Minimum posts per topic in the training sample when corpus permits.
_FLOOR_PER_TOPIC = 10

# Ceiling for scraper.scrape() per build_corpus call.
_MAX_SCRAPE_POSTS = 10_000


async def build_corpus(
    *,
    scraper: Any,
    source_url: str,
    window_months: int = 12,
    n_train: int = 200,
    n_judge_per_topic: int = 10,
    n_manual_per_topic: int = 5,
    seed: int | None = 42,
    min_cluster: int = 5,
) -> dict[str, Any]:
    """Scrape, filter, cluster, and sample a training + eval corpus.

    Args:
        scraper:           Object with ``scrape(url, limit_months, max_posts)`` coroutine.
        source_url:        Feed URL for the persona.
        window_months:     How many months back to scrape.
        n_train:           Target training set size (topic-balanced).
        n_judge_per_topic: LLM-judge eval holdout size per topic.
        n_manual_per_topic: Manual eval holdout size per topic.
        seed:              RNG seed for reproducible shuffles.
        min_cluster:       Minimum posts per HDBSCAN cluster (default 5).

    Returns:
        ``{"training": list[SocialPost],
           "eval_judge": dict[str, list[SocialPost]],
           "eval_manual": dict[str, list[SocialPost]]}``

    Raises:
        SystemExit(5) if training set is empty after scrape + injection filter.
    """
    # ------------------------------------------------------------------
    # 1. Scrape
    # ------------------------------------------------------------------
    all_posts: list[SocialPost] = await scraper.scrape(
        source_url, limit_months=window_months, max_posts=_MAX_SCRAPE_POSTS
    )

    # ------------------------------------------------------------------
    # 2. Injection filter — drop matches; emit warn envelope per drop
    # ------------------------------------------------------------------
    filtered: list[SocialPost] = []
    for post in all_posts:
        if is_likely_injection(post.text):
            emit(
                level="warn",
                stage="scrape_filter",
                code="injection_filter_match",
                message="Post dropped: injection pattern detected.",
                drops=[{
                    "source": post.source_url or str(id(post)),
                    "reason": "injection_filter_match",
                }],
            )
        else:
            filtered.append(post)

    if not filtered:
        emit(
            level="warn",
            stage="corpus_sampler",
            code="empty_corpus",
            message="No posts survived injection filter; cannot build training corpus.",
        )
        sys.exit(5)

    # ------------------------------------------------------------------
    # 3. Cluster
    # ------------------------------------------------------------------
    texts = [p.text for p in filtered]
    labels, centroids = cluster_posts(texts, seed=seed, min_cluster_size=min_cluster)

    # Map topic label → list of SocialPost (same order as filtered)
    by_topic: dict[str, list[SocialPost]] = defaultdict(list)
    for post, label in zip(filtered, labels):
        by_topic[label].append(post)

    # ------------------------------------------------------------------
    # 4. Per-topic shuffle → eval holdout (eval first, disjoint from train)
    # ------------------------------------------------------------------
    rng = random.Random(seed)

    eval_judge: dict[str, list[SocialPost]] = {}
    eval_manual: dict[str, list[SocialPost]] = {}
    training_pool: dict[str, list[SocialPost]] = {}

    for topic, posts in by_topic.items():
        shuffled = list(posts)
        rng.shuffle(shuffled)

        # Eval slices come first so they are never in the training pool.
        judge_slice = shuffled[:n_judge_per_topic]
        manual_slice = shuffled[n_judge_per_topic: n_judge_per_topic + n_manual_per_topic]
        remainder = shuffled[n_judge_per_topic + n_manual_per_topic:]

        eval_judge[topic] = judge_slice
        eval_manual[topic] = manual_slice
        training_pool[topic] = remainder

    # ------------------------------------------------------------------
    # 5. Topic-balanced training sample
    # ------------------------------------------------------------------
    training = _sample_balanced(training_pool, n_train, rng)

    # Build parallel topic label list for training posts.
    # by_topic maps topic → list[SocialPost]; build id → topic for O(1) lookup.
    post_to_topic: dict[int, str] = {}
    for topic, posts in by_topic.items():
        for post in posts:
            post_to_topic[id(post)] = topic
    training_topic_labels: list[str] = [
        post_to_topic.get(id(p), "_global") for p in training
    ]

    return {
        "training": training,
        "training_topic_labels": training_topic_labels,
        "topic_centroids": centroids,
        "eval_judge": eval_judge,
        "eval_manual": eval_manual,
    }


def _sample_balanced(
    pool: dict[str, list[Any]],
    n_total: int,
    rng: random.Random,
) -> list[Any]:
    """Sample ``n_total`` posts balanced across topics.

    Strategy:
    1. Allocate floor=_FLOOR_PER_TOPIC to each topic where pool permits.
    2. Distribute remaining quota proportionally.
    3. If total available < n_total, return all available posts.

    The result is a flat list (order: round-robin across topics by
    allocation, stable within each topic by the pre-shuffled pool order).
    """
    topics = list(pool.keys())
    if not topics:
        return []

    available = {t: list(pool[t]) for t in topics}
    total_available = sum(len(v) for v in available.values())

    if total_available == 0:
        return []

    n_target = min(n_total, total_available)

    # --- Step 1: floor allocation ---
    alloc: dict[str, int] = {}
    for t in topics:
        alloc[t] = min(_FLOOR_PER_TOPIC, len(available[t]))

    floor_used = sum(alloc.values())
    remaining_quota = n_target - floor_used

    if remaining_quota > 0:
        # --- Step 2: proportional top-up ---
        # Only topics that still have posts beyond their floor allocation.
        extras = {t: max(0, len(available[t]) - alloc[t]) for t in topics}
        total_extras = sum(extras.values())

        if total_extras > 0:
            for t in topics:
                prop = extras[t] / total_extras
                add = int(prop * remaining_quota)
                alloc[t] += min(add, extras[t])

        # Correct any rounding shortfall greedily
        still_remaining = n_target - sum(alloc.values())
        for t in topics:
            if still_remaining <= 0:
                break
            headroom = len(available[t]) - alloc[t]
            if headroom > 0:
                give = min(headroom, still_remaining)
                alloc[t] += give
                still_remaining -= give

    # --- Step 3: collect in round-robin order ---
    result: list[Any] = []
    for t in topics:
        result.extend(available[t][: alloc[t]])

    # Shuffle the final result with the same RNG for reproducibility
    rng.shuffle(result)
    return result
