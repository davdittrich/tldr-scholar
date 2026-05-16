#!/usr/bin/env python3
"""Utility to interactively refine a persona profile using atomic probes."""
import argparse
import sys
from pathlib import Path

import yaml

try:
    from gemini_acp import summarize_via_gemini, ACP_AVAILABLE
except ImportError:
    summarize_via_gemini = None
    ACP_AVAILABLE = False

PROBING_PROMPT = """\
You are an expert persona designer. You are helping a user refine their AI writing persona.
Based on the existing profile and the user's answers to your probing cases, 
provide an updated, more nuanced YAML profile.

Existing Profile:
{existing_yaml}

Probing Results:
{probing_results}

Return ONLY the updated YAML block with deep intent fields and confidence scores.
"""

AMBIGUITY_RESOLVER_PROMPT = """\
The user's persona profile has an ambiguity in the attribute: {attribute}.
Confidence Score: {score}%

Create a side-by-side "Resolution Case":
1. Present a hypothetical source claim related to {attribute}.
2. Ask the user how their persona would specifically re-author this claim 
   to resolve the ambiguity or contradiction found in their corpus.

Return only the probe text.
"""

GAP_PROBE_PROMPT = """\
The user's persona profile has a substantive gap: it is silent on {topic}.
This topic is critical to the domain of {domain}.

Create a "Worldview Extrapolation Case":
1. Present a major domain conclusion regarding {topic}.
2. Ask the user how their persona (given their worldview) would reframe or 
   suppress this conclusion.

Return only the probe text.
"""


def main():
    parser = argparse.ArgumentParser(description="Interactively refine a persona profile.")
    parser.add_argument("name", help="Name of the persona to refine")
    args = parser.parse_args()

    config_dir = Path.home() / ".config" / "tldr-scholar" / "personas"
    persona_path = config_dir / f"{args.name}.yaml"

    if not persona_path.exists():
        print(f"Error: Persona '{args.name}' not found at {persona_path}")
        sys.exit(1)

    existing_yaml = persona_path.read_text()
    try:
        data = yaml.safe_load(existing_yaml)
    except Exception:
        print("Error: Invalid YAML profile.")
        sys.exit(1)

    if summarize_via_gemini is None or not ACP_AVAILABLE:
        print("Error: gemini-acp not installed.")
        sys.exit(1)

    print(f"Refining deep persona: {args.name}")
    print("-" * 40)

    # Identify flags for refinement
    confidence = data.get("attribute_confidence", {})
    flags = []
    for attr, score in confidence.items():
        if score < 70:
            flags.append({"type": "ambiguity", "attr": attr, "score": score})
    
    # Simple gap detection (mocked for now, requires domain baseline in real implementation)
    if "revelation_priorities" not in data or not data["revelation_priorities"]:
        flags.append({"type": "gap", "topic": "General Domain Debates", "domain": data.get("role", "General")})

    if not flags:
        print("Profile is high-confidence. No gaps detected.")
        sys.exit(0)

    results = []
    for i, flag in enumerate(flags, 1):
        if flag["type"] == "ambiguity":
            prompt = AMBIGUITY_RESOLVER_PROMPT.format(attribute=flag["attr"], score=flag["score"])
        else:
            prompt = GAP_PROBE_PROMPT.format(topic=flag["topic"], domain=flag["domain"])
            
        print(f"\n[{i}/{len(flags)}] Generating probe case...")
        probe, _ = summarize_via_gemini(text="", prompt=prompt)
        print("\n" + probe.strip())
        ans = input("\nYour Resolution > ").strip()
        if ans:
            results.append(f"Probe: {probe.strip()}\nUser Resolution: {ans}")

    if not results:
        print("\nNo refinement provided. Exiting.")
        sys.exit(0)

    # Final Synthesis
    print("\nSynthesizing refined deep profile...")
    final_prompt = PROBING_PROMPT.format(
        existing_yaml=existing_yaml,
        probing_results="\n\n".join(results)
    )
    refined_yaml, _ = summarize_via_gemini(text="", prompt=final_prompt)

    clean_result = refined_yaml.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        data = yaml.safe_load(clean_result)
        with open(persona_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"\nSuccess! Deep persona '{args.name}' updated and refined.")
    except Exception as e:
        print(f"Error parsing refined YAML: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
