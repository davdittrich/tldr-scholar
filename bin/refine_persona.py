#!/usr/bin/env python3
"""Utility to interactively refine a persona profile."""
import argparse
import sys
from pathlib import Path

import yaml

try:
    from gemini_acp import summarize_via_gemini, ACP_AVAILABLE
except ImportError:
    summarize_via_gemini = None
    ACP_AVAILABLE = False

REFine_PROMPT = """\
You are an expert persona designer. You are helping a user refine their AI writing persona.
Based on the existing profile below and the user's answers to your interview questions, 
provide an updated, more nuanced YAML profile.

Existing Profile:
{existing_yaml}

User's Additional Context:
{user_feedback}

Return ONLY the updated YAML block. Include fields for:
- agenda
- worldview
- extraction_filter
- persuasion_goal
- linguistic_nuances (Any specific phrases, idioms, or quirks)
"""

INTERVIEW_QUESTIONS = [
    "What is the primary mission or 'agenda' of your writing?",
    "What is your implied worldview or philosophical leaning?",
    "What specific information do you prioritize extracting from a text (e.g., power dynamics, data rigor)?",
    "What do you explicitly ignore or leave out?",
    "What is the long-term goal of your persuasion? What do you want to convince readers of?",
    "Are there specific linguistic nuances or 'catchphrases' that make your writing identifiable?"
]

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
    
    print(f"Refining persona: {args.name}")
    print("-" * 40)
    
    answers = []
    for i, q in enumerate(INTERVIEW_QUESTIONS, 1):
        print(f"\n[{i}/{len(INTERVIEW_QUESTIONS)}] {q}")
        ans = input("> ").strip()
        if ans:
            answers.append(f"Q: {q}\nA: {ans}")

    if not answers:
        print("\nNo feedback provided. Exiting.")
        sys.exit(0)

    user_feedback = "\n\n".join(answers)

    if summarize_via_gemini is None or not ACP_AVAILABLE:
        print("\nError: gemini-acp not installed. Cannot use LLM for refinement.")
        print("Here is your feedback collected so you can update the YAML manually:")
        print(user_feedback)
        sys.exit(1)

    prompt = REFine_PROMPT.format(existing_yaml=existing_yaml, user_feedback=user_feedback)
    print("\nGenerating refined profile via Gemini...")
    result, _ = summarize_via_gemini(text="", prompt=prompt)

    if not result:
        print("Error: Gemini failed to produce a refined profile.")
        sys.exit(1)

    # Basic cleanup
    clean_result = result.strip()
    if "```yaml" in clean_result:
        clean_result = clean_result.split("```yaml")[1].split("```")[0].strip()
    elif "```" in clean_result:
        clean_result = clean_result.split("```")[1].split("```")[0].strip()

    try:
        data = yaml.safe_load(clean_result)
        if not isinstance(data, dict):
            print("Error: LLM output is not a valid YAML dictionary.")
            sys.exit(1)

        with open(persona_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
            
        print(f"\nSuccess! Refined persona '{args.name}' saved to {persona_path}")
    except Exception as e:
        print(f"Error parsing or saving YAML: {e}")
        print("Raw output from Gemini:")
        print(clean_result)
        sys.exit(1)

if __name__ == "__main__":
    main()
