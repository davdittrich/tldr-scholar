#!/usr/bin/env python3
"""Synthesize a writing style profile with stratified engagement sampling.

CLI flags:
    source              feed URL (Mastodon/Bluesky) or local samples file
    --name NAME         persona output name (writes to DEFAULT_PERSONA_DIR/<name>.yaml)
    --months N          lookback window in months (default 12)
    --max-posts N       sample size cap (default 200)
    --concurrency N     parallel link fetches (default 5)
    --skip-links        skip article ingestion, use post bodies only (default off)
    --full-baselines    run all 3 baselines per source (claims+extractive+abstractive);
                        default is claims-only to minimise LLM cost
"""
import argparse
import hashlib
import json
import sys
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
import httpx
from loguru import logger

from tldr_scholar.config import DEFAULT_PERSONA_DIR
from tldr_scholar.corpus_cache import CorpusCache
from tldr_scholar.corpus_sampler import build_corpus
from tldr_scholar.ingest import ingest
from tldr_scholar.scrapers import ScraperFactory, SocialPost, UnknownURLError
from tldr_scholar.ingestion_engine import LinkIngester
from tldr_scholar.prompts import DEEP_SYNTHESIS_PROMPT
from tldr_scholar.source_baseline import build_baselines
from tldr_scholar.correlator import correlate_against_baselines
from tldr_scholar.error_contract import emit_envelope, emit_drop_summary, EXIT_CODES
from tldr_scholar.preflight import check_embedding_model_cached
from tldr_scholar.aggregator import aggregate_topic, aggregate_global
from tldr_scholar.personas import DeltaRecord, Persona, TopicProfile, write_persona_yaml
from tldr_scholar.topic_cluster import EMBEDDING_MODEL_NAME

try:
    from gemini_acp import summarize_via_gemini, ACP_AVAILABLE
except ImportError:
    summarize_via_gemini = None
    ACP_AVAILABLE = False

# Per-stage corpus cache root
CACHE_ROOT = Path.home() / ".cache" / "tldr-scholar" / "stages"

# Prompts
CLASSIFICATION_PROMPT = """\
Analyze the following list of social media posts.
Group them into logical "Substantive Domains" (e.g. Economics, Tech, Politics, Ethics).

Posts:
{posts}

Return ONLY a YAML dictionary mapping "domain_name" to a list of post indices (0-indexed).
"""

async def call_gemini(prompt: str, label: str) -> Any:
    logger.debug(f"[GEMINI CALL: {label}]")
    logger.debug("-" * 20)
    logger.debug(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
    logger.debug("-" * 20)
    
    result, _ = summarize_via_gemini(text="", prompt=prompt, timeout=180)
    if not result:
        return None
        
    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        return yaml.safe_load(clean_result)
    except yaml.YAMLError as e:
        logger.error(f"YAML Parse Error in {label}: {e}")
        return None

def _llm_caller():
    """Return an async callable(prompt: str) -> str for source_baseline / correlator."""
    async def _call(prompt: str) -> str:
        result, _ = summarize_via_gemini(text="", prompt=prompt, timeout=180)
        return result or ""
    return _call

async def classify_domains(posts: list[SocialPost]) -> dict[str, list[int]]:
    # Batch classification for up to 200 posts (Mastodon limit approx)
    limit = 200
    post_texts = "\n".join([f"{i}: {p.text[:150]}" for i, p in enumerate(posts[:limit])])
    prompt = CLASSIFICATION_PROMPT.format(posts=post_texts)
    data = await call_gemini(prompt, "Classification")
    return data if isinstance(data, dict) else {}

async def synthesize_deep_profile(reports: list[dict]) -> dict:
    reports_yaml = yaml.dump(reports)
    prompt = DEEP_SYNTHESIS_PROMPT.format(reports=reports_yaml)
    data = await call_gemini(prompt, "Synthesis")
    return data if isinstance(data, dict) else {}

async def run_synthesis(args):
    # Pre-flight: warn if embedding model not cached (non-fatal)
    check_embedding_model_cached()

    if not ACP_AVAILABLE:
        emit_envelope(
            level="error",
            stage="startup",
            code="acp_unavailable",
            message="gemini_acp module unavailable; install gemini-acp package",
        )
        sys.exit(EXIT_CODES["internal"])

    # Stage cache — always initialised so write-through works on every run.
    cache = CorpusCache(root=CACHE_ROOT)

    # Fingerprint of CLI flags that affect cluster/correlate/aggregate outputs.
    flag_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "full_baselines": getattr(args, "full_baselines", False),
                "min_cluster": getattr(args, "min_cluster", 5),
                "window_months": getattr(args, "window_months", 12),
                "n_train": getattr(args, "n_train", 200),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    # --reset: wipe this stage + all downstream caches BEFORE sampling
    if getattr(args, "reset", None):
        cache.invalidate(args.reset)
        emit_envelope(
            level="info",
            stage="cache",
            code="cache_invalidated",
            message=f"--reset={args.reset} cascaded.",
        )

    async with httpx.AsyncClient(follow_redirects=True) as client:
        source_str = str(args.source)
        if ":/" in source_str and "://" not in source_str: source_str = source_str.replace(":/", "://")
        if ":///" in source_str: source_str = source_str.replace(":///", "://")
        
        try:
            scraper = ScraperFactory.get_scraper(source_str, client)
        except UnknownURLError as e:
            logger.error(f"URL not supported: {e}")
            sys.exit(1)

        # 1-3. Scrape -> injection-filter -> cluster -> topic-balanced sample
        #
        # Cluster cache: keyed by source_url + embedding_model + flag_fingerprint.
        # corpus_hash is not available yet (it requires the scraped posts), so the
        # cluster key uses only the stable inputs that precede scraping.
        logger.info(f"Building corpus: scrape {args.window_months}m window, n_train={args.n_train}...")
        _cluster_key = {
            "source_url": source_str,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "flag_fingerprint": flag_fingerprint,
        }
        _cluster_cached = None
        try:
            _cluster_cached = cache.get("cluster", _cluster_key)
        except Exception as _e:
            logger.warning(f"[cache] cluster GET failed (continuing fresh): {_e}")
        if _cluster_cached is not None:
            logger.info("[cache] cluster HIT — skipping build_corpus")
            sampled_posts = [SocialPost(**p) for p in _cluster_cached["posts"]]
            training_topic_labels = _cluster_cached["labels"]
            topic_centroids = _cluster_cached["centroids"]
        else:
            corpus_result = await build_corpus(
                scraper=scraper,
                source_url=source_str,
                window_months=args.window_months,
                n_train=args.n_train,
                n_judge_per_topic=args.n_judge_per_topic,
                n_manual_per_topic=args.n_manual_per_topic,
                seed=42,
                min_cluster=getattr(args, "min_cluster", 5),
            )
            sampled_posts: list[SocialPost] = corpus_result["training"]
            training_topic_labels: list[str] = corpus_result.get("training_topic_labels", [])
            topic_centroids: dict[str, list[float]] = corpus_result.get("topic_centroids", {})
            try:
                cache.put("cluster", _cluster_key, {
                    "labels": training_topic_labels,
                    "centroids": topic_centroids,
                    "posts": [p.model_dump(mode="json") for p in sampled_posts],
                })
            except Exception as _e:
                logger.warning(f"[cache] cluster PUT failed (non-fatal): {_e}")
        logger.info(
            f"Corpus ready: {len(sampled_posts)} training posts."
        )

        # corpus_hash: computed from the filtered training posts (stable across re-runs
        # with same source + flags).  Used as part of the correlate + aggregate keys.
        corpus_hash = hashlib.sha256(
            "\n".join(sorted(p.text for p in sampled_posts)).encode()
        ).hexdigest()[:16]

        # 4. Ingest Links for samples
        corpus = []
        if args.skip_links:
            logger.info("--skip-links set: skipping article ingestion, using post bodies only")
            for post in sampled_posts:
                corpus.append((post.text, post.text))
        else:
            ingester = LinkIngester(concurrency=args.concurrency)
            articles = await ingester.process_posts(sampled_posts)
            for art in articles:
                source = art.body if art.body else art.post.text
                corpus.append((source, art.post.text))

    # 5. Atomic Pipeline — correlate each (source, post) pair against baselines.
    #
    # Correlate cache: keyed by source_url + embedding_model + corpus_hash + flag_fingerprint.
    caller = _llm_caller()
    _corr_key = {
        "source_url": source_str,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "corpus_hash": corpus_hash,
        "flag_fingerprint": flag_fingerprint,
    }
    _corr_cached = None
    try:
        _corr_cached = cache.get("correlate", _corr_key)
    except Exception as _e:
        logger.warning(f"[cache] correlate GET failed (continuing fresh): {_e}")
    if _corr_cached is not None:
        logger.info("[cache] correlate HIT — skipping correlation loop")
        final_reports = [DeltaRecord(**d) for d in _corr_cached["reports"]]
        report_topic_labels = _corr_cached["topic_labels"]
        report_post_texts = _corr_cached["post_texts"]
    else:
        final_reports = []
        # Parallel to corpus: topic label and post text per entry (for grouping)
        report_topic_labels: list[str] = []
        report_post_texts: list[str] = []
        for i, (source_text, post_text) in enumerate(corpus):
            topic_label = training_topic_labels[i] if i < len(training_topic_labels) else "_global"
            logger.info(f">>> Analyzing Pair {i+1}/{len(corpus)}")
            baselines = await build_baselines(
                source_text,
                llm_call=caller,
                full=getattr(args, "full_baselines", False),
            )
            if baselines.claims is None and baselines.extractive_summary is None and baselines.abstractive_summary is None:
                emit_envelope(
                    level="warn",
                    stage="source_baseline",
                    code="all_baselines_failed",
                    message="All 3 baselines failed for source; dropping from corpus.",
                    drops=[{"source": f"pair_{i+1}"}],
                )
                continue
            delta_records = await correlate_against_baselines(
                post_text,
                baselines,
                llm_call=caller,
            )
            if delta_records:
                final_reports.extend(delta_records)
                for _ in delta_records:
                    report_topic_labels.append(topic_label)
                    report_post_texts.append(post_text)
            else:
                logger.debug(f"Skipping pair {i+1}: all correlations returned no delta")
        try:
            cache.put("correlate", _corr_key, {
                "reports": [d.model_dump(mode="json") for d in final_reports],
                "topic_labels": report_topic_labels,
                "post_texts": report_post_texts,
            })
        except Exception as _e:
            logger.warning(f"[cache] correlate PUT failed (non-fatal): {_e}")

    # 6. Per-topic and global aggregation
    # Group final_reports by topic label
    topic_to_deltas: dict[str, list[DeltaRecord]] = defaultdict(list)
    topic_to_posts: dict[str, list[str]] = defaultdict(list)
    for delta, tlabel, ptext in zip(final_reports, report_topic_labels, report_post_texts):
        topic_to_deltas[tlabel].append(delta)
        if ptext not in topic_to_posts[tlabel]:
            topic_to_posts[tlabel].append(ptext)

    # Also collect posts-only from sampled_posts that had no delta records
    # (to ensure posts field is populated from full sampled set)
    seen_post_texts: set[str] = set()
    for posts_list in topic_to_posts.values():
        seen_post_texts.update(posts_list)
    for i, post in enumerate(sampled_posts):
        tlabel = training_topic_labels[i] if i < len(training_topic_labels) else "_global"
        if post.text not in seen_post_texts:
            topic_to_posts[tlabel].append(post.text)
            seen_post_texts.add(post.text)

    # Compute output path early so exhaustion handlers can write partial YAML.
    persona_name = args.name or "persona"
    output_path = DEFAULT_PERSONA_DIR / f"{persona_name}.yaml"
    incomplete_stages: list[str] = []

    # Aggregate cache: keyed by source_url + embedding_model + corpus_hash + flag_fingerprint.
    # Contains both per-topic profiles and global fields so a single GET can skip both
    # aggregate_topic and aggregate_global.
    _agg_key = {
        "source_url": source_str,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "corpus_hash": corpus_hash,
        "flag_fingerprint": flag_fingerprint,
    }
    _agg_cached = None
    try:
        _agg_cached = cache.get("aggregate", _agg_key)
    except Exception as _e:
        logger.warning(f"[cache] aggregate GET failed (continuing fresh): {_e}")
    if _agg_cached is not None:
        logger.info("[cache] aggregate HIT — skipping per-topic and global aggregation")
        topics = {k: TopicProfile(**v) for k, v in _agg_cached["topics"].items()}
        global_fields = _agg_cached["global_fields"]
    else:
        # Build TopicProfile per topic
        topics: dict[str, TopicProfile] = {}
        topic_ok: list[bool] = []
        for tlabel in set(list(topic_to_deltas.keys()) + list(topic_to_posts.keys())):
            centroid = topic_centroids.get(tlabel, [])
            posts_for_topic = list(topic_to_posts.get(tlabel, []))
            deltas_for_topic = list(topic_to_deltas.get(tlabel, []))
            tp, ok = await aggregate_topic(
                label=tlabel,
                centroid=centroid,
                posts=posts_for_topic,
                deltas=deltas_for_topic,
                llm_call=caller,
            )
            topics[tlabel] = tp
            topic_ok.append(ok)

        # Propagate per-topic failures to persona status (tldr-scholar-58f.5).
        # aggregate_topic returns (TopicProfile, bool); False means degraded/fallback.
        if topic_ok and not all(topic_ok):
            incomplete_stages.append("aggregate_topic:partial")

        # If no topics at all, create a _global fallback so Persona.topics is non-empty
        if not topics:
            topics["_global"] = TopicProfile(
                label="_global",
                centroid=[],
                sample_size=0,
                posts=[p.text for p in sampled_posts],
            )

        # Global synthesis
        global_fields, global_ok = await aggregate_global(final_reports, caller)

        # aggregate_global returns (dict, success); signal failure if parse failed
        if not global_ok:
            incomplete_stages.append("aggregate_global:failed")

        try:
            cache.put("aggregate", _agg_key, {
                "topics": {k: v.model_dump(mode="json") for k, v in topics.items()},
                "global_fields": global_fields,
            })
        except Exception as _e:
            logger.warning(f"[cache] aggregate PUT failed (non-fatal): {_e}")

    # Build Persona
    persona = Persona(
        name=persona_name,
        embedding_model=EMBEDDING_MODEL_NAME,
        status="incomplete" if incomplete_stages else "complete",
        incomplete_stages=incomplete_stages,
        topics=topics,
        agenda=global_fields.get("agenda", ""),
        worldview=global_fields.get("worldview", ""),
        pivot_logic=global_fields.get("pivot_logic", ""),
        identifiable_nuances=global_fields.get("identifiable_nuances", []),
    )

    write_persona_yaml(persona, output_path)
    logger.info(f"Success! {output_path}")
    emit_drop_summary()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--name")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--max-posts", type=int, default=200)
    parser.add_argument("--window-months", dest="window_months", type=int, default=12,
                        help="Scrape window in months (default: 12).")
    parser.add_argument("--n-train", dest="n_train", type=int, default=200,
                        help="Training set size (default: 200).")
    parser.add_argument("--n-judge-per-topic", dest="n_judge_per_topic", type=int, default=10,
                        help="LLM-judge eval holdout per topic (default: 10).")
    parser.add_argument("--n-manual-per-topic", dest="n_manual_per_topic", type=int, default=5,
                        help="Manual eval holdout per topic (default: 5).")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--skip-links", action="store_true", help="Skip article ingestion; use post bodies only.")
    parser.add_argument("--full-baselines", dest="full_baselines", action="store_true", default=False,
                        help="Run all 3 baselines (claims+extractive+abstractive). Default: claims only.")
    parser.add_argument(
        "--reset",
        choices=["cluster", "correlate", "aggregate", "all"],
        default=None,
        dest="reset",
        help="Wipe this stage and all downstream caches before running.",
    )
    parser.add_argument(
        "--min-cluster",
        dest="min_cluster",
        type=int,
        default=5,
        help="Minimum posts per HDBSCAN cluster (default: 5).",
    )
    args = parser.parse_args()
    asyncio.run(run_synthesis(args))

if __name__ == "__main__":
    main()
