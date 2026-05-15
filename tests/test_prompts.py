"""Tests for tldr_scholar.prompts."""
from __future__ import annotations

import pytest

from tldr_scholar.models import AudienceEnum, ToneEnum
from tldr_scholar.prompts import PromptBuilder


class TestPromptBuilder:
    def test_get_audience_instruction_expert(self):
        builder = PromptBuilder()
        instr = builder._get_audience_instruction(AudienceEnum.EXPERT)
        assert "precise technical terminology" in instr
        assert "methodology nuances" in instr

    def test_get_audience_instruction_layman(self):
        builder = PromptBuilder()
        instr = builder._get_audience_instruction(AudienceEnum.LAYMAN)
        assert "simple analogies" in instr
        assert "academic jargon" in instr

    def test_get_scientific_structure_expert(self):
        builder = PromptBuilder()
        # Expert mode should have sentences 2-4 focused on data/method
        structure = builder._get_scientific_structure(5, AudienceEnum.EXPERT)
        assert "Sentence 3: The broad methodology" in structure
        assert "Sentence 4: The primary, most significant result" in structure

    def test_get_scientific_structure_layman(self):
        builder = PromptBuilder()
        # Layman mode should focus on problem and general findings
        structure = builder._get_scientific_structure(5, AudienceEnum.LAYMAN)
        assert "simple question" in structure
        assert "important discovery" in structure

    def test_get_tone_instruction_casual(self):
        builder = PromptBuilder()
        instr = builder._get_tone_instruction(ToneEnum.CASUAL)
        assert "conversational" in instr or "approachable" in instr

    def test_build_system_prompt_scientific(self):
        builder = PromptBuilder()
        prompt = builder.build_system_prompt(
            mode="scientific",
            max_chars=500,
            focus="test",
            hashtag_instruction="hashtags",
            sentence_count=5,
            audience=AudienceEnum.LAYMAN,
            tone=ToneEnum.PROFESSIONAL,
        )
        assert "simple analogies" in prompt
        assert "test" in prompt
        assert "hashtags" in prompt

    def test_build_system_prompt_general(self):
        builder = PromptBuilder()
        prompt = builder.build_system_prompt(
            mode="general",
            max_chars=500,
            focus="test",
            hashtag_instruction="hashtags",
            sentence_count=5,
            audience=AudienceEnum.EXPERT,
            tone=ToneEnum.ANALYTICAL,
        )
        assert "analytical" in prompt.lower()
        assert "test" in prompt
