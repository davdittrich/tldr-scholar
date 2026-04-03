"""Prompt templates for summarization backends."""
from __future__ import annotations

# Sentence counts by length preset
SENTENCE_COUNTS = {"short": 3, "medium": 5, "long": 7}

# ---------------------------------------------------------------------------
# Scientific mode (default) — IMRAD-aware structured summarization
# ---------------------------------------------------------------------------

SCIENTIFIC_SYSTEM_PROMPT = """\
You are a scientific article summarizer.

Analysis Instructions:
Do not weigh the text equally. Prioritize extracting information from the \
Title, Abstract, Conclusion, Introduction, and Results. Skim the Methods \
only for broad context. Identify the core IMRAD elements: Context \
(Introduction), Approach (Methods), Key Findings (Results), and Impact \
(Discussion).

Output Format:
Provide exactly a {sentence_count}-sentence summary in approximately \
{max_chars} characters, written in plain, simple language. Explain the \
concepts as if teaching someone completely outside the field. Strip away \
all dense academic jargon, complex formulas, and niche terminology, \
translating the core ideas into clear, everyday language.

{sentence_structure}

{focus_instruction}

{hashtag_instruction}

Verification Guardrail:
Before finalizing your output, compare your draft against the original text. \
Verify that: (1) No outside knowledge, external facts, or unsupported claims \
have been introduced. (2) The core findings, methods, and limitations \
accurately reflect the authors' explicit statements, not assumptions. \
If your draft fails this check, revise the inaccurate sentences before \
providing the final output."""

_SCIENTIFIC_SENTENCES = {
    3: (
        "Structure the sentences as follows:\n"
        "Sentence 1: The general background and the specific problem being addressed.\n"
        "Sentence 2: The methodology and the primary, most significant result.\n"
        "Sentence 3: The broader implication of the findings, including limitations."
    ),
    5: (
        "Structure the sentences exactly as follows:\n"
        "Sentence 1: The general background of the topic.\n"
        "Sentence 2: The specific problem or knowledge gap the authors are addressing.\n"
        "Sentence 3: The broad methodology used to approach the problem.\n"
        "Sentence 4: The primary, most significant result or data point uncovered.\n"
        "Sentence 5: The broader implication of the findings, including a brief "
        "mention of the study's limitations."
    ),
    7: (
        "Structure the sentences as follows:\n"
        "Sentence 1: The general background of the topic.\n"
        "Sentence 2: The specific problem or knowledge gap being addressed.\n"
        "Sentence 3: Why this problem matters or what prior work has missed.\n"
        "Sentence 4: The broad methodology used to approach the problem.\n"
        "Sentence 5: The primary, most significant result or data point.\n"
        "Sentence 6: Secondary results or supporting evidence.\n"
        "Sentence 7: The broader implication, including limitations and future directions."
    ),
}

# ---------------------------------------------------------------------------
# General mode — for non-academic text (blog posts, news, documents)
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """\
Summarize the following document in approximately {max_chars} characters \
using exactly {sentence_count} sentences.
Focus on: {focus}.
Be concise, precise, and factual. Do not add information not in the source.

{hashtag_instruction}"""

# ---------------------------------------------------------------------------
# Single-prompt template (Gemini / Ollama — not chat-based)
# ---------------------------------------------------------------------------

SINGLE_PROMPT_TEMPLATE = """\
{system_prompt}

<document>
{text}
</document>"""


def build_system_prompt(
    mode: str,
    max_chars: int,
    focus: str,
    hashtag_instruction: str,
    sentence_count: int = 5,
) -> str:
    """Build the system prompt for the given mode."""
    if mode == "scientific":
        sentence_structure = _SCIENTIFIC_SENTENCES.get(
            sentence_count,
            _SCIENTIFIC_SENTENCES[5],  # fallback to medium
        )
        focus_instruction = (
            f"Thematic focus: {focus}."
            if focus and focus != "main findings and novel insights"
            else ""
        )
        return SCIENTIFIC_SYSTEM_PROMPT.format(
            max_chars=max_chars,
            sentence_count=sentence_count,
            sentence_structure=sentence_structure,
            focus_instruction=focus_instruction,
            hashtag_instruction=hashtag_instruction,
        ).strip()
    else:
        return GENERAL_SYSTEM_PROMPT.format(
            max_chars=max_chars,
            sentence_count=sentence_count,
            focus=focus,
            hashtag_instruction=hashtag_instruction,
        ).strip()


def build_single_prompt(
    text: str,
    mode: str,
    max_chars: int,
    focus: str,
    hashtag_instruction: str,
    sentence_count: int = 5,
) -> str:
    """Build the full prompt for single-prompt APIs (Gemini, Ollama).

    Wraps the text in <document> delimiters.
    """
    system = build_system_prompt(mode, max_chars, focus, hashtag_instruction, sentence_count)
    return SINGLE_PROMPT_TEMPLATE.format(system_prompt=system, text=text)
