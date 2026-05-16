#!/usr/bin/env python3
"""Utility to synthesize a writing style profile from a corpus of posts."""
import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

from tldr_scholar.config import DEFAULT_PERSONA_DIR
from tldr_scholar.ingest import ingest

try:
    from gemini_acp import summarize_via_gemini, ACP_AVAILABLE
except ImportError:
    summarize_via_gemini = None
    ACP_AVAILABLE = False

DECOMPOSITION_PROMPT = """\
Analyze the following text and decompose it into a list of atomic statements, 
claims, and conclusions.

Text:
{text}

Return ONLY a YAML list of objects with 'id' and 'claim' fields.
"""

CORRELATION_PROMPT = """\
Compare the user's social media post against the following atomic statements 
from the source text.

Statements:
{statements}

User Post:
{post_text}

For each statement, determine if it was:
1. shared: The post revealed this claim.
2. suppressed: The post ignored this claim (despite it being substantive).
3. pivoted: The post transformed this claim into a different argument.

Return ONLY a YAML list of objects with 'statement_id', 'status', and 'intent'.
"""

SYNTHESIS_PROMPT = """\
Based on the following corpus of writing samples and source documents, 
synthesize a deep cognitive architecture for an AI writing assistant.

Corpus:
{text}

Focus on:
- agenda: High-level purpose of the persona's writing.
- worldview: Implied philosophical/political leaning.
- revelation_priorities: What to prioritize.
- suppression_rules: What to ignore or filter.
- substantive_anchors: Core recurring topics or frames.
- pivot_logic: How claims are reframed.
- rhetorical_strategy: Linguistic patterns.
- identifiable_nuances: Unique stylistic 'fingerprints'.

Return ONLY a YAML dictionary.
"""

DEEP_SYNTHESIS_PROMPT = """\
Synthesize a global persona profile from the following atomic delta reports.

Delta Reports:
{reports}

Focus on identifying recurring intents and systematic revelation/suppression patterns.
Quantify your confidence (0-100) for each major attribute.

Return ONLY a YAML dictionary with 'profile' and 'confidence' keys.
"""


def decompose_source(text: str) -> list[dict]:
    """Decompose source text into atomic statements via LLM."""
    if summarize_via_gemini is None or not ACP_AVAILABLE:
        return []

    prompt = DECOMPOSITION_PROMPT.format(text=text)
    result, _ = summarize_via_gemini(text="", prompt=prompt)
    if not result:
        return []

    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        data = yaml.safe_load(clean_result)
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and 'id' not in item:
                    item['id'] = item.get('statement_id', f'c{i+1}')
            return data
        return []
    except Exception:
        return []


def correlate_post_to_source(statements: list[dict], post_text: str) -> list[dict]:
    """Map user post against atomic statements to find deltas."""
    if summarize_via_gemini is None or not ACP_AVAILABLE:
        return []

    statements_yaml = yaml.dump(statements)
    prompt = CORRELATION_PROMPT.format(statements=statements_yaml, post_text=post_text)
    result, _ = summarize_via_gemini(text="", prompt=prompt)
    if not result:
        return []

    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        data = yaml.safe_load(clean_result)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def synthesize_deep_profile(reports: list[list[dict]]) -> dict:
    """Synthesize global rules and confidence scores from atomic deltas."""
    if summarize_via_gemini is None or not ACP_AVAILABLE:
        return {}

    reports_yaml = yaml.dump(reports)
    prompt = DEEP_SYNTHESIS_PROMPT.format(reports=reports_yaml)
    result, _ = summarize_via_gemini(text="", prompt=prompt)
    if not result:
        return {}

    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        data = yaml.safe_load(clean_result)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Synthesize writing style from corpus.")
    parser.add_argument("source", type=Path, help="Path to text or JSONL corpus")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--name", help="Name for the persona")
    parser.add_argument("--output", type=Path, help="Output YAML path")
    args = parser.parse_args()

    if not args.source.exists():
        logger.error(f"Source file {args.source} not found")
        sys.exit(1)

    if summarize_via_gemini is None or not ACP_AVAILABLE:
        logger.error("gemini-acp not installed or available. Cannot perform analysis.")
        sys.exit(1)

    data = {}
    if args.format == "jsonl":
        reports = []
        logger.info(f"Executing bottom-up atomic pipeline for corpus: {args.source}")
        with open(args.source, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    post = entry.get("post")
                    url = entry.get("url")
                    if not post or not url:
                        continue
                    
                    logger.info(f"Processing atomic delta for URL: {url}")
                    source_text, _ = ingest(url)
                    if not source_text:
                        continue
                        
                    statements = decompose_source(source_text)
                    if not statements:
                        continue
                        
                    delta = correlate_post_to_source(statements, post)
                    if delta:
                        reports.append(delta)
                except Exception as e:
                    logger.warning(f"Skipping malformed corpus entry: {e}")
                    continue
        
        if not reports:
            logger.error("No valid delta reports generated from corpus.")
            sys.exit(1)
            
        logger.info("Synthesizing cognitive architecture from atomic reports...")
        synth_data = synthesize_deep_profile(reports)
        if not synth_data or not isinstance(synth_data, dict):
            logger.error("Failed to synthesize deep profile.")
            sys.exit(1)
        
        data = synth_data.get("profile", {})
        if not isinstance(data, dict):
            data = {}
        data["attribute_confidence"] = synth_data.get("confidence", {})

    else:
        # Enforce Atomic Pipeline as Default for text
        text = args.source.read_text()
        logger.info(f"Executing atomic pipeline for text source: {args.source}")
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        reports = []
        for i, chunk in enumerate(chunks[:10]):
            logger.info(f"Processing text partition {i+1}/{min(10, len(chunks))}...")
            statements = decompose_source(chunk)
            if not statements: continue
            delta = correlate_post_to_source(statements, chunk)
            if delta: reports.append(delta)
            
        if not reports:
            logger.error("No valid delta reports generated.")
            sys.exit(1)
            
        synth_data = synthesize_deep_profile(reports)
        if not synth_data:
            sys.exit(1)
        data = synth_data.get("profile", {})
        if not isinstance(data, dict):
            data = {}
        data["attribute_confidence"] = synth_data.get("confidence", {})

    if args.name:
        data["name"] = args.name

    name = data.get("name", args.source.stem) if isinstance(data, dict) else args.source.stem
    output_path = args.output or DEFAULT_PERSONA_DIR / f"{name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
        
    logger.info(f"Success! Deep Persona '{name}' saved to {output_path}")


if __name__ == "__main__":
    main()
