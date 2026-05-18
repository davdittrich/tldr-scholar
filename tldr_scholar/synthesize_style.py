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
        sys.exit(1)

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
        logger.info(f"Building corpus: scrape {args.window_months}m window, n_train={args.n_train}...")
        corpus_result = await build_corpus(
            scraper=scraper,
            source_url=source_str,
            window_months=args.window_months,
            n_train=args.n_train,
            n_judge_per_topic=args.n_judge_per_topic,
            n_manual_per_topic=args.n_manual_per_topic,
            seed=42,
        )
        sampled_posts: list[SocialPost] = corpus_result["training"]
        training_topic_labels: list[str] = corpus_result.get("training_topic_labels", [])
        topic_centroids: dict[str, list[float]] = corpus_result.get("topic_centroids", {})
        logger.info(
            f"Corpus ready: {len(sampled_posts)} training posts; "
            f"{sum(len(v) for v in corpus_result['eval_judge'].values())} judge-eval; "
            f"{sum(len(v) for v in corpus_result['eval_manual'].values())} manual-eval."
        )

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

    # 5. Atomic Pipeline
    final_reports = []
    # Parallel to corpus: topic label and post text per entry (for grouping)
    report_topic_labels: list[str] = []
    report_post_texts: list[str] = []
    caller = _llm_caller()
    for i, (source_text, post_text) in enumerate(corpus):
        topic_label = training_topic_labels[i] if i < len(training_topic_labels) else "_global"
        logger.info(f">>> Analyzing Pair {i+1}/{len(corpus)}")
        baselines = await build_baselines(
            source_text,
            full=getattr(args, "full_baselines", False),
            llm_call=caller,
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

    # Build TopicProfile per topic
    topics: dict[str, TopicProfile] = {}
    try:
        for tlabel in set(list(topic_to_deltas.keys()) + list(topic_to_posts.keys())):
            centroid = topic_centroids.get(tlabel, [])
            posts_for_topic = list(topic_to_posts.get(tlabel, []))
            deltas_for_topic = list(topic_to_deltas.get(tlabel, []))
            tp = await aggregate_topic(
                label=tlabel,
                centroid=centroid,
                posts=posts_for_topic,
                deltas=deltas_for_topic,
                llm_call=caller,
            )
            topics[tlabel] = tp
    except Exception:  # stage boundary: LLM API exhaustion
        incomplete_stages.append("aggregate_topic")
        partial = Persona(
            name=persona_name,
            embedding_model=EMBEDDING_MODEL_NAME,
            status="incomplete",
            incomplete_stages=incomplete_stages,
            topics=topics or {"_global": TopicProfile(
                label="_global", centroid=[], sample_size=0,
                posts=[p.text for p in sampled_posts],
            )},
        )
        write_persona_yaml(partial, output_path)
        emit_envelope(
            level="error",
            stage="aggregate_topic",
            code="llm_exhausted",
            message="LLM API exhausted during per-topic aggregation; partial persona written.",
        )
        emit_drop_summary()
        sys.exit(EXIT_CODES["llm_exhausted"])

    # If no topics at all, create a _global fallback so Persona.topics is non-empty
    if not topics:
        topics["_global"] = TopicProfile(
            label="_global",
            centroid=[],
            sample_size=0,
            posts=[p.text for p in sampled_posts],
        )

    # Global synthesis
    try:
        global_fields = await aggregate_global(final_reports, caller)
    except Exception:  # stage boundary: LLM API exhaustion
        incomplete_stages.append("aggregate_global")
        partial = Persona(
            name=persona_name,
            embedding_model=EMBEDDING_MODEL_NAME,
            status="incomplete",
            incomplete_stages=incomplete_stages,
            topics=topics,
        )
        write_persona_yaml(partial, output_path)
        emit_envelope(
            level="error",
            stage="aggregate_global",
            code="llm_exhausted",
            message="LLM API exhausted during global synthesis; partial persona written.",
        )
        emit_drop_summary()
        sys.exit(EXIT_CODES["llm_exhausted"])

    # Build Persona
    persona = Persona(
        name=persona_name,
        embedding_model=EMBEDDING_MODEL_NAME,
        status="complete",
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
    args = parser.parse_args()
    asyncio.run(run_synthesis(args))

if __name__ == "__main__":
    main()
