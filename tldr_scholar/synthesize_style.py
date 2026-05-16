#!/usr/bin/env python3
"""Utility to synthesize a deep writing style persona from text samples."""
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

SYNTHESIS_PROMPT = """\
Analyze the writing style AND underlying intent of the following text samples. 
Provide a detailed writing style profile in YAML format with the following fields:

- name: A short identifier (lowercase, no spaces)
- role: A brief description of the persona (e.g., "Academic Economist")
- tone: 2-3 adjectives describing the voice (e.g., "analytical, objective")
- structure_pattern: "stitched" or "bullet_points"
- hashtag_style: "lowercase" or "pascal"

# Deep Intent Fields (Crucial)
- agenda: What is the primary purpose or "mission" of this author's writing?
- worldview: What is the author's implied philosophical or political leaning?
- revelation_priorities: What substantive arguments/data points does this author amplify?
- suppression_rules: What significant statements in a source does this author intentionally ignore?
- substantive_anchors: What core evidence types does this author rely on?
- pivot_logic: How are source claims re-indexed into the worldview?
- rhetorical_strategy: How does this persona build an argument?
- identifiable_nuances: List of linguistic quirks/idioms.

Return ONLY the YAML block.

Samples:
{text}
"""

DECOMPOSITION_PROMPT = """\
Analyze the following substantive sections of a research paper or article 
(Introduction, Discussion, Conclusion). 

Decompose the text into its constituent "Atomic Statements"—distinct claims, 
empirical findings, or major conclusions.

For each statement, provide:
- id: A unique short ID (e.g., claim_1)
- content: The substantive text of the claim.
- section: Which section it originated from (introduction, discussion, conclusion).

Return ONLY a YAML list of these objects.

Text:
{text}
"""

CORRELATION_PROMPT = """\
Compare the following user social media post against a list of "Atomic Statements" 
from the source material the user is sharing.

Identify which statements were:
- shared: Directly mentioned or summarized.
- suppressed: Significant source claims the user ignored.
- pivoted: Claims the user transformed or re-authored into their own worldview.

For each statement, provide:
- statement_id: The ID of the atomic statement.
- status: shared, suppressed, or pivoted.
- intent: Inferred local intent (why was this shared/suppressed/pivoted?).

Return ONLY a YAML list of these correlation objects.

Atomic Statements:
{statements}

User Post:
{post_text}
"""

DEEP_SYNTHESIS_PROMPT = """\
Analyze the following collection of Atomic Delta Reports. Each report 
represents the differential between a substantive source and a user's post.

Synthesize a comprehensive "Cognitive Architecture" for this persona.
Identify global rules for revelation, suppression, and pivoting.

Provide a detailed YAML profile with:
- profile:
    agenda: Primary mission.
    worldview: Implied philosophical/political leaning.
    revelation_priorities: List of substantive arguments/data points amplified.
    suppression_rules: Content intentionally ignored or deemed deceptive.
    substantive_anchors: Core evidence types relied on.
    pivot_logic: How source claims are re-indexed into the worldview.
    rhetorical_strategy: How the persona builds an argument.
    identifiable_nuances: List of linguistic quirks/idioms.
- confidence:
    Assign a confidence score (0-100) to each of the above fields based 
    on evidence consistency across reports.

Return ONLY the YAML block.

Delta Reports:
{reports}
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
    parser = argparse.ArgumentParser(description="Synthesize deep writing style from samples.")
    parser.add_argument("source", type=Path, help="File containing text samples or JSONL corpus")
    parser.add_argument("--name", help="Override persona name")
    parser.add_argument("--output", type=Path, help="Output YAML path")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text", 
                        help="Input format (text for raw samples, jsonl for post+url pairs)")
    args = parser.parse_args()

    if not args.source.exists():
        logger.error(f"Source file {args.source} not found")
        sys.exit(1)

    if summarize_via_gemini is None or not ACP_AVAILABLE:
        logger.error("gemini-acp not installed or available. Cannot perform analysis.")
        sys.exit(1)

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
                    # Fetch substantive sections (mocking full IMRAD fetch for now, using ingest)
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
        if not synth_data:
            logger.error("Failed to synthesize deep profile.")
            sys.exit(1)
        
        # Merge profile and confidence into flat dict for YAML
        data = synth_data.get("profile", {})
        data["attribute_confidence"] = synth_data.get("confidence", {})
        if args.name:
            data["name"] = args.name

    else:
        # Legacy/Shallow single-pass synthesis
        text = args.source.read_text()
        logger.info("Analyzing deep style and intent via single-pass synthesis...")
        prompt = SYNTHESIS_PROMPT.format(text=text)
        result, _ = summarize_via_gemini(text="", prompt=prompt)
        
        if not result:
            logger.error("Gemini failed to produce a profile.")
            sys.exit(1)

        clean_result = result.strip()
        if "```yaml" in clean_result:
            clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
        elif "```" in clean_result:
            clean_result = clean_result.split("```")[1].split("```")[0].strip()

        try:
            data = yaml.safe_load(clean_result)
            if not isinstance(data, dict):
                logger.error(f"LLM output is not a valid YAML dictionary.")
                sys.exit(1)
            if args.name:
                data["name"] = args.name
        except Exception as e:
            logger.error(f"Error parsing YAML: {e}")
            sys.exit(1)

    # Save logic
    name = data.get("name", args.source.stem)
    output_path = args.output or DEFAULT_PERSONA_DIR / f"{name}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
        
    logger.info(f"Success! Deep Persona '{name}' saved to {output_path}")


if __name__ == "__main__":
    main()
