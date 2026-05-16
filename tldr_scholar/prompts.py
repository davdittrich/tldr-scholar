"""Prompt templates for summarization backends."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tldr_scholar.types import AudienceEnum, ToneEnum
from tldr_scholar.personas import PersonaManager

if TYPE_CHECKING:
    pass

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
{max_chars} characters. {audience_instruction}

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
    "expert": {
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
    },
    "layman": {
        3: (
            "Structure the sentences as follows:\n"
            "Sentence 1: What the study is about and what big question it answers.\n"
            "Sentence 2: The main discovery and why it matters in simple terms.\n"
            "Sentence 3: What this means for the future and what we still don't know."
        ),
        5: (
            "Structure the sentences exactly as follows:\n"
            "Sentence 1: The general topic of the research.\n"
            "Sentence 2: The simple question the researchers wanted to answer.\n"
            "Sentence 3: How they did the study in broad strokes.\n"
            "Sentence 4: The most important discovery they made.\n"
            "Sentence 5: Why this is important for regular people and any major warnings."
        ),
        7: (
            "Structure the sentences as follows:\n"
            "Sentence 1: The big picture topic.\n"
            "Sentence 2: The specific mystery the authors wanted to solve.\n"
            "Sentence 3: Why regular people should care about this mystery.\n"
            "Sentence 4: A simple explanation of how they studied it.\n"
            "Sentence 5: The coolest or most useful thing they found.\n"
            "Sentence 6: Other interesting bits they discovered.\n"
            "Sentence 7: What this means for the world and what comes next."
        ),
    }
}

# Fallback/Default structure (same as expert)
_SCIENTIFIC_SENTENCES["student"] = _SCIENTIFIC_SENTENCES["expert"]

# ---------------------------------------------------------------------------
# General mode — for non-academic text (blog posts, news, documents)
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """\
Summarize the following document in approximately {max_chars} characters \
using exactly {sentence_count} sentences.
{tone_instruction}
Focus on: {focus}.
Be concise, precise, and factual. Do not add information not in the source.

{hashtag_instruction}"""

# ---------------------------------------------------------------------------
# Persona Template (Generic)
# ---------------------------------------------------------------------------

PERSONA_SYSTEM_PROMPT = """\
You are {role}.
Summarize the following document with maximum information density and a {tone} tone.
Target length: {sentence_count} sentences, approximately {max_chars} characters.
Avoid all hype, emotional language, and corporate jargon.

{deep_intent_instruction}

{linguistic_nuance_instruction}

Output Format:
{pattern_instruction}

{hashtag_instruction}
"""

_PATTERNS = {
    "stitched": (
        "1. Start directly with the title and source: [Title]{source_line}\n"
        "2. Use the 'Stitched Quotes' pattern: select the most critical 3-5 fragments "
        "from the text and stitch them together using ellipses exactly like this: "
        "\"… [fragment] … [fragment] …\".\n"
        "3. Finish with a brief analytical synthesis sentence."
    ),
    "bullet_points": (
        "1. Start with the [Title].\n"
        "2. Provide the core findings as a list of 3-5 concise bullet points.\n"
        "3. Focus on empirical evidence and statistical significance."
    )
}

# ---------------------------------------------------------------------------
# Single-prompt template (Gemini / Ollama — not chat-based)
# ---------------------------------------------------------------------------

SINGLE_PROMPT_TEMPLATE = """\
{system_prompt}

<document>
{text}
</document>"""


class PromptBuilder:
    """Class to build customized system and user prompts."""

    def __init__(self):
        self._persona_manager = PersonaManager()

    def _get_audience_instruction(self, audience: AudienceEnum) -> str:
        """Get instructions for the specific audience persona."""
        if audience == AudienceEnum.EXPERT:
            return (
                "Write in plain, simple language but use precise technical terminology. "
                "Focus on methodology nuances and specific data points. Explain the concepts "
                "as if teaching someone with a PhD in this field."
            )
        elif audience == AudienceEnum.LAYMAN:
            return (
                "Explain the concepts using simple analogies and clear, everyday language. "
                "Avoid all dense academic jargon and niche terminology. Focus on the 'big picture' "
                "impact as if teaching someone completely outside the field."
            )
        elif audience == AudienceEnum.STUDENT:
            return (
                "Explain the concepts clearly, defining any necessary technical terms. "
                "Focus on the core logic and findings. Translate complex ideas into clear language "
                "suitable for a student or non-expert with some interest."
            )
        return ""

    def _get_tone_instruction(self, tone: ToneEnum) -> str:
        """Get instructions for the specific tone."""
        if tone == ToneEnum.CASUAL:
            return "Use a conversational and approachable tone."
        elif tone == ToneEnum.ANALYTICAL:
            return "Use a critical and analytical tone, focusing on logic and evidence."
        elif tone == ToneEnum.PROFESSIONAL:
            return "Use a professional, neutral, and objective tone."
        return ""

    def _get_scientific_structure(self, sentence_count: int, audience: AudienceEnum) -> str:
        """Get the IMRAD structure instructions based on audience and length."""
        # Determine persona key
        if audience == AudienceEnum.LAYMAN:
            persona = "layman"
        else:
            persona = "expert"
            
        persona_structures = _SCIENTIFIC_SENTENCES.get(persona, _SCIENTIFIC_SENTENCES["expert"])
        return persona_structures.get(sentence_count, persona_structures[5])

    def build_system_prompt(
        self,
        mode: str,
        max_chars: int,
        focus: str,
        hashtag_instruction: str,
        sentence_count: int = 5,
        audience: AudienceEnum | None = None,
        tone: ToneEnum | None = None,
        persona: str | None = None,
        text: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Build the system prompt for the given mode and audience."""
        # Set defaults if not provided
        if audience is None:
            audience = AudienceEnum.EXPERT
        if tone is None:
            tone = ToneEnum.PROFESSIONAL

        # Persona override
        if persona:
            p_config = self._persona_manager.get_persona(persona)
            if p_config:
                # Word count guard: if input < 300 words, fall back to expert
                word_count = len(re.findall(r"\w+", text))
                if word_count >= 300:
                    source = (metadata or {}).get("source", "")
                    source_line = f" [{source}]" if source and source.startswith("http") else ""
                    
                    pattern_template = _PATTERNS.get(
                        p_config.structure_pattern, 
                        _PATTERNS["stitched"]
                    )
                    pattern_instr = pattern_template.format(source_line=source_line)
                    
                    # Build Deep Intent instructions
                    intent_parts = []
                    if p_config.agenda:
                        intent_parts.append(f"Your writing agenda: {p_config.agenda}")
                    if p_config.worldview:
                        intent_parts.append(f"Your implied worldview/leaning: {p_config.worldview}")
                    if p_config.revelation_priorities:
                        priorities = ", ".join(p_config.revelation_priorities)
                        intent_parts.append(f"REVEAL and amplify these substantive arguments: {priorities}")
                    if p_config.suppression_rules:
                        rules = ", ".join(p_config.suppression_rules)
                        intent_parts.append(f"SUPPRESS and ignore these deceptive or noisy claims: {rules}")
                    if p_config.pivot_logic:
                        intent_parts.append(f"Substantive Re-authoring (Pivot Logic): {p_config.pivot_logic}")
                    if p_config.rhetorical_strategy:
                        intent_parts.append(f"Rhetorical strategy: {p_config.rhetorical_strategy}")
                    
                    deep_intent_instr = "\n".join(intent_parts)
                    
                    # Linguistic nuances
                    nuance_instr = ""
                    if p_config.identifiable_nuances:
                        nuances = ", ".join(p_config.identifiable_nuances)
                        nuance_instr = f"Mandatory identifiable nuances (fingerprint): {nuances}"
                    
                    return PERSONA_SYSTEM_PROMPT.format(
                        role=p_config.role,
                        tone=p_config.tone,
                        max_chars=max_chars,
                        sentence_count=sentence_count,
                        deep_intent_instruction=deep_intent_instr,
                        linguistic_nuance_instruction=nuance_instr,
                        pattern_instruction=pattern_instr,
                        hashtag_instruction=hashtag_instruction,
                    ).strip()

        if mode == "scientific":
            sentence_structure = self._get_scientific_structure(sentence_count, audience)
            focus_instruction = (
                f"Thematic focus: {focus}."
                if focus and focus != "main findings and novel insights"
                else ""
            )
            audience_instr = self._get_audience_instruction(audience)
            
            return SCIENTIFIC_SYSTEM_PROMPT.format(
                max_chars=max_chars,
                sentence_count=sentence_count,
                audience_instruction=audience_instr,
                sentence_structure=sentence_structure,
                focus_instruction=focus_instruction,
                hashtag_instruction=hashtag_instruction,
            ).strip()
        else:
            tone_instr = self._get_tone_instruction(tone)
            return GENERAL_SYSTEM_PROMPT.format(
                max_chars=max_chars,
                sentence_count=sentence_count,
                tone_instruction=tone_instr,
                focus=focus,
                hashtag_instruction=hashtag_instruction,
            ).strip()

    def build_single_prompt(
        self,
        text: str,
        mode: str,
        max_chars: int,
        focus: str,
        hashtag_instruction: str,
        sentence_count: int = 5,
        audience: AudienceEnum | None = None,
        tone: ToneEnum | None = None,
        persona: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Build the full prompt for single-prompt APIs (Gemini, Ollama)."""
        system = self.build_system_prompt(
            mode, max_chars, focus, hashtag_instruction, sentence_count, 
            audience, tone, persona, text, metadata
        )
        return SINGLE_PROMPT_TEMPLATE.format(system_prompt=system, text=text)


# ---------------------------------------------------------------------------
# Legacy functional interface for backward compatibility
# ---------------------------------------------------------------------------

def build_system_prompt(*args, **kwargs) -> str:
    """Backward compatible wrapper for PromptBuilder.build_system_prompt."""
    return PromptBuilder().build_system_prompt(*args, **kwargs)


def build_single_prompt(*args, **kwargs) -> str:
    """Backward compatible wrapper for PromptBuilder.build_single_prompt."""
    return PromptBuilder().build_single_prompt(*args, **kwargs)
