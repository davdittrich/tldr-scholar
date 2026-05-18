#!/usr/bin/env python3
"""Synthesize a writing style profile with stratified engagement sampling.

CLI flags:
    source              feed URL (Mastodon/Bluesky) or local samples file
    --name NAME         persona output name (writes to DEFAULT_PERSONA_DIR/<name>.yaml)
    --months N          lookback window in months (default 12)
    --max-posts N       sample size cap (default 200)
    --concurrency N     parallel link fetches (default 5)
    --skip-links        skip article ingestion, use post bodies only (default off)
"""
import argparse
import sys
import asyncio
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
from tldr_scholar.prompts import DECOMPOSITION_PROMPT, CORRELATION_PROMPT, DEEP_SYNTHESIS_PROMPT
from tldr_scholar.preflight import check_embedding_model_cached

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

async def classify_domains(posts: list[SocialPost]) -> dict[str, list[int]]:
    # Batch classification for up to 200 posts (Mastodon limit approx)
    limit = 200
    post_texts = "\n".join([f"{i}: {p.text[:150]}" for i, p in enumerate(posts[:limit])])
    prompt = CLASSIFICATION_PROMPT.format(posts=post_texts)
    data = await call_gemini(prompt, "Classification")
    return data if isinstance(data, dict) else {}

async def decompose_source(text: str) -> list[dict]:
    prompt = DECOMPOSITION_PROMPT.format(text=text)
    data = await call_gemini(prompt, "Decomposition")
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'id' not in item:
                item['id'] = item.get('statement_id', f'c{i+1}')
        return data
    return []

async def correlate_post_to_source(statements: list[dict], post_text: str) -> list[dict]:
    statements_yaml = yaml.dump(statements)
    prompt = CORRELATION_PROMPT.format(statements=statements_yaml, post_text=post_text)
    data = await call_gemini(prompt, "Correlation")
    return data if isinstance(data, list) else []

async def synthesize_deep_profile(reports: list[list[dict]]) -> dict:
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
    for i, (source_text, post_text) in enumerate(corpus):
        logger.info(f">>> Analyzing Pair {i+1}/{len(corpus)}")
        statements = await decompose_source(source_text)
        if not statements:
            logger.debug(f"Skipping pair {i+1}: decompose_source returned no statements")
            continue
        delta = await correlate_post_to_source(statements, post_text)
        if delta:
            final_reports.append(delta)
        else:
            logger.debug(f"Skipping pair {i+1}: correlate_post_to_source returned no delta")

    # 6. Synthesis
    synth_data = await synthesize_deep_profile(final_reports)
    data = synth_data.get("profile", {})
    data["attribute_confidence"] = synth_data.get("confidence", {})
    if args.name: data["name"] = args.name

    name = data.get("name", "persona")
    output_path = DEFAULT_PERSONA_DIR / f"{name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    logger.info(f"Success! {output_path}")

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
    args = parser.parse_args()
    asyncio.run(run_synthesis(args))

if __name__ == "__main__":
    main()
