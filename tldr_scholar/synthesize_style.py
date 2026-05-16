#!/usr/bin/env python3
"""Utility to synthesize a deep writing style persona from text samples."""
import argparse
import sys
from pathlib import Path

import yaml

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
- extraction_filter: What specific types of information does this author prioritize 
  (e.g., p-values, funding sources, power dynamics) and what do they explicitly ignore?
- persuasion_goal: What is the author trying to convince their readers of in the long run?

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


def main():
    parser = argparse.ArgumentParser(description="Synthesize deep writing style from samples.")
    parser.add_argument("source", type=Path, help="File containing text samples")
    parser.add_argument("--name", help="Override persona name")
    parser.add_argument("--output", type=Path, help="Output YAML path")
...