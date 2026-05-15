#!/usr/bin/env python3
"""Utility to synthesize a writing style persona from text samples."""
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
Analyze the writing style of the following text samples. 
Provide a concise writing style profile in YAML format with the following fields:
- name: A short identifier (lowercase, no spaces)
- role: A brief description of the persona (e.g., "Academic Economist")
- tone: 2-3 adjectives describing the voice (e.g., "analytical, objective")
- structure_pattern: Instructions for sentence structure and formatting
- hashtag_style: "lowercase" or "pascal"

Return ONLY the YAML block.

Samples:
{text}
"""

def main():
    parser = argparse.ArgumentParser(description="Synthesize writing style from samples.")
    parser.add_argument("source", type=Path, help="File containing text samples")
    parser.add_argument("--name", help="Override persona name")
    parser.add_argument("--output", type=Path, help="Output YAML path")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: {args.source} not found")
        sys.exit(1)

    text = args.source.read_text()
    
    if summarize_via_gemini is None or not ACP_AVAILABLE:
        print("Error: gemini-acp not installed or available. Cannot perform analysis.")
        sys.exit(1)

    prompt = SYNTHESIS_PROMPT.format(text=text)
    print("Analyzing style via Gemini...")
    result, _ = summarize_via_gemini(text="", prompt=prompt)
    
    if not result:
        print("Error: Gemini failed to produce a profile.")
        sys.exit(1)

    # Basic cleanup if model included triple backticks
    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        data = yaml.safe_load(clean_result)
        if not isinstance(data, dict):
            print("Error: LLM output is not a valid YAML dictionary.")
            print(f"Output received: {clean_result}")
            sys.exit(1)

        if args.name:
            data["name"] = args.name
            
        name = data.get("name", args.source.stem)
        output_path = args.output
        if not output_path:
            config_dir = Path.home() / ".config" / "tldr-scholar" / "personas"
            config_dir.mkdir(parents=True, exist_ok=True)
            output_path = config_dir / f"{name}.yaml"

        with open(output_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
            
        print(f"Success! Persona '{name}' saved to {output_path}")
    except Exception as e:
        print(f"Error parsing or saving YAML: {e}")
        print("Raw output from Gemini:")
        print(clean_result)
        sys.exit(1)

if __name__ == "__main__":
    main()
