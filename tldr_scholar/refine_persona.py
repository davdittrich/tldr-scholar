#!/usr/bin/env python3
"""Utility to interactively refine a persona profile using atomic probes."""
import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from tldr_scholar.config import DEFAULT_PERSONA_DIR

try:
    from gemini_acp import summarize_via_gemini, ACP_AVAILABLE
except ImportError:
    summarize_via_gemini = None
    ACP_AVAILABLE = False

PROBING_PROMPT = """\
You are an expert persona designer. You are helping a user refine their AI writing persona.
Based on the existing profile and the user's answers to your probing cases, 
provide an updated, more nuanced writing style profile.

Existing Profile:
{existing_yaml}

Probing Results:
{probing_results}

Focus specifically on filling gaps in:
- agenda
- worldview
- revelation_priorities
- suppression_rules
- pivot_logic
- rhetorical_strategy
- identifiable_nuances

Update the confidence scores where the user's resolution provided higher certainty.

Return ONLY the updated YAML block.
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

DOMAIN_GAP_DETECTION_PROMPT = """\
Analyze the following persona profile for an AI writing assistant.
Identify substantive "Gaps" where the persona's corpus is likely silent on 
critical domain-specific debates or methodologies.

Persona Role: {role}
Current Stance Space: {stance_space}

Compare this against a "Standard Expert Map" for this domain. 
Provide a list of up to 3 "Missing Topics" (substantive gaps) that should be 
probed to complete the persona's cognitive architecture.

Return ONLY a YAML list of these missing topics.
"""


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Deep merge update into base dictionary."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def detect_profile_gaps(data: dict[str, Any]) -> list[str]:
    """Use LLM to detect substantive gaps in the persona profile."""
    if summarize_via_gemini is None or not ACP_AVAILABLE:
        return []
    
    role = data.get("role", "General Researcher")
    stance_space = yaml.dump({
        "agenda": data.get("agenda"),
        "revelation_priorities": data.get("revelation_priorities"),
        "suppression_rules": data.get("suppression_rules")
    })
    
    prompt = DOMAIN_GAP_DETECTION_PROMPT.format(role=role, stance_space=stance_space)
    result, _ = summarize_via_gemini(text="", prompt=prompt)
    if not result:
        return []
        
    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        gaps = yaml.safe_load(clean_result)
        if isinstance(gaps, list):
            return [str(g) for g in gaps]
        return []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Interactively refine a persona profile.")
    parser.add_argument("name", help="Name of the persona to refine")
    args = parser.parse_args()

    persona_path = DEFAULT_PERSONA_DIR / f"{args.name}.yaml"

    if not persona_path.exists():
        logger.error(f"Persona '{args.name}' not found at {persona_path}")
        sys.exit(1)

    existing_text = persona_path.read_text()
    try:
        existing_data = yaml.safe_load(existing_text)
        if not isinstance(existing_data, dict):
            raise ValueError("Persona YAML is not a dictionary")
    except Exception as e:
        logger.error(f"Invalid YAML profile at {persona_path}: {e}")
        sys.exit(1)

    if summarize_via_gemini is None or not ACP_AVAILABLE:
        logger.error("gemini-acp not installed.")
        sys.exit(1)

    logger.info(f"Refining deep persona: {args.name}")
    print("-" * 40)

    # Identify flags for refinement
    confidence = existing_data.get("attribute_confidence", {})
    flags = []
    
    # 1. Logical Ambiguities (Low Confidence)
    for attr, score in confidence.items():
        if score < 70:
            flags.append({"type": "ambiguity", "attr": attr, "score": score})
    
    # 2. Substantive Gaps (Domain Boundary Comparison)
    logger.info("Detecting substantive profile gaps via domain boundary comparison...")
    gaps = detect_profile_gaps(existing_data)
    for gap in gaps:
        flags.append({"type": "gap", "topic": gap, "domain": existing_data.get("role", "General")})

    if not flags:
        logger.info("Profile is high-confidence. No gaps detected.")
        sys.exit(0)

    results = []
    for i, flag in enumerate(flags, 1):
        if flag["type"] == "ambiguity":
            prompt = AMBIGUITY_RESOLVER_PROMPT.format(attribute=flag["attr"], score=flag["score"])
        else:
            prompt = GAP_PROBE_PROMPT.format(topic=flag["topic"], domain=flag["domain"])
            
        logger.info(f"[{i}/{len(flags)}] Generating probe case for: {flag.get('attr') or flag.get('topic')}...")
        probe, _ = summarize_via_gemini(text="", prompt=prompt)
        if not probe:
            logger.warning("Gemini failed to generate a probe case.")
            continue
            
        print("\n" + probe.strip())
        ans = input("\nYour Resolution > ").strip()
        if ans:
            results.append(f"Probe: {probe.strip()}\nUser Resolution: {ans}")

    if not results:
        logger.info("No refinement provided. Exiting.")
        sys.exit(0)

    # Final Synthesis
    logger.info("Synthesizing refined deep profile...")
    final_prompt = PROBING_PROMPT.format(
        existing_yaml=existing_text,
        probing_results="\n\n".join(results)
    )
    refined_yaml_text, _ = summarize_via_gemini(text="", prompt=final_prompt)

    if not refined_yaml_text:
        logger.error("Gemini failed to produce a refined profile.")
        sys.exit(1)

    clean_result = refined_yaml_text.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        new_data = yaml.safe_load(clean_result)
        if not isinstance(new_data, dict):
            logger.error("LLM output is not a valid YAML dictionary.")
            sys.exit(1)
            
        # Perform deep merge to prevent data loss of fields LLM might omit
        updated_data = deep_merge(existing_data, new_data)
            
        with open(persona_path, "w") as f:
            yaml.dump(updated_data, f, sort_keys=False)
        logger.info(f"Success! Deep persona '{args.name}' updated and refined.")
    except Exception as e:
        logger.error(f"Error parsing refined YAML: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
