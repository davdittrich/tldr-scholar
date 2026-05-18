"""Tests for tldr_scholar.prompts (v2 persona schema)."""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path
from tldr_scholar.models import AudienceEnum, ToneEnum
from tldr_scholar.prompts import PromptBuilder, DECOMPOSITION_PROMPT, CORRELATION_PROMPT, DEEP_SYNTHESIS_PROMPT


def _v2_persona(name: str, **topic_kwargs) -> dict:
    """Build a minimal v2 persona dict with one topic."""
    topic = {
        "label": "_global",
        "centroid": [0.0] * 384,
        "sample_size": 10,
        "revelation_priorities": topic_kwargs.get("revelation_priorities", []),
        "suppression_rules": topic_kwargs.get("suppression_rules", []),
        "substantive_anchors": topic_kwargs.get("substantive_anchors", []),
        "rhetorical_strategy": topic_kwargs.get("rhetorical_strategy", ""),
    }
    return {
        "name": name,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "role": topic_kwargs.get("role", "analyst"),
        "tone": topic_kwargs.get("tone", "analytical"),
        "structure_pattern": topic_kwargs.get("structure_pattern", "stitched"),
        "hashtag_style": "lowercase",
        "agenda": topic_kwargs.get("agenda", ""),
        "worldview": topic_kwargs.get("worldview", ""),
        "topics": {"_global": topic},
    }


class TestPromptBuilder:
    def test_get_audience_instruction_expert(self):
        builder = PromptBuilder()
        instr = builder._get_audience_instruction(AudienceEnum.EXPERT)
        assert "precise technical terminology" in instr

    def test_build_system_prompt_persona(self, tmp_path):
        """Basic persona path: role + tone + structure_pattern → uses PERSONA_SYSTEM_PROMPT."""
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        data = _v2_persona("test", role="academic", tone="sharp", structure_pattern="stitched")
        with open(persona_dir / "test.yaml", "w") as f:
            yaml.dump(data, f)

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        long_text = "word " * 400
        prompt = builder.build_system_prompt(
            mode="scientific", max_chars=500, focus="", hashtag_instruction="",
            persona="test", text=long_text, metadata={"source": "http://example.com"}
        )

        assert "academic" in prompt
        assert "Stitched Quotes" in prompt
        assert "[http://example.com]" in prompt
        assert "5 sentences" in prompt
        assert "500 characters" in prompt

    def test_build_system_prompt_deep_intent(self, tmp_path):
        """Persona with agenda/worldview/revelation_priorities → injected into prompt."""
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        data = _v2_persona(
            "deep",
            role="skeptic",
            tone="analytical",
            structure_pattern="stitched",
            agenda="Expose capture",
            worldview="Critical",
            revelation_priorities=["funding", "p-values"],
        )
        with open(persona_dir / "deep.yaml", "w") as f:
            yaml.dump(data, f)

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        long_text = "word " * 400
        prompt = builder.build_system_prompt(
            mode="scientific", max_chars=500, focus="", hashtag_instruction="",
            persona="deep", text=long_text
        )

        assert "Expose capture" in prompt
        assert "Critical" in prompt
        assert "REVEAL" in prompt
        assert "funding" in prompt

    def test_build_system_prompt_bullet_pattern(self, tmp_path):
        """bullet_points structure_pattern renders bullet points instructions."""
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        data = _v2_persona("bullet", role="data scientist", tone="concise", structure_pattern="bullet_points")
        with open(persona_dir / "bullet.yaml", "w") as f:
            yaml.dump(data, f)

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        long_text = "word " * 400
        prompt = builder.build_system_prompt(
            mode="scientific", max_chars=500, focus="", hashtag_instruction="",
            persona="bullet", text=long_text
        )

        assert "bullet points" in prompt
        assert "empirical evidence" in prompt
        assert "Stitched Quotes" not in prompt

    def test_persona_short_text_guard(self, tmp_path):
        """Texts < 300 words fall back to expert scientific prompt."""
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        data = _v2_persona("test", role="r", tone="t", structure_pattern="s")
        with open(persona_dir / "test.yaml", "w") as f:
            yaml.dump(data, f)

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        short_text = "word " * 100
        prompt = builder.build_system_prompt(
            mode="scientific", max_chars=500, focus="", hashtag_instruction="",
            persona="test", text=short_text
        )

        assert "scientific article summarizer" in prompt
        assert "Stitched Quotes" not in prompt

    def test_persona_missing_url_guard(self, tmp_path):
        """Non-URL source metadata is not injected into the source_line."""
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        data = _v2_persona("test", role="r", tone="t", structure_pattern="stitched")
        with open(persona_dir / "test.yaml", "w") as f:
            yaml.dump(data, f)

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        long_text = "word " * 400
        prompt = builder.build_system_prompt(
            mode="scientific", max_chars=500, focus="", hashtag_instruction="",
            persona="test", text=long_text, metadata={"source": "local.pdf"}
        )

        assert "[Title]" in prompt
        assert "[local.pdf]" not in prompt


class TestPromptsRelocation:
    """Verify prompts from synthesize_style are now resident in prompts.py."""

    def test_decomposition_prompt_in_prompts_module(self):
        assert DECOMPOSITION_PROMPT is not None
        assert "{text}" in DECOMPOSITION_PROMPT
        assert len(DECOMPOSITION_PROMPT) > 50

    def test_correlation_prompt_in_prompts_module(self):
        assert CORRELATION_PROMPT is not None
        assert "{statements}" in CORRELATION_PROMPT
        assert "{post_text}" in CORRELATION_PROMPT

    def test_deep_synthesis_prompt_in_prompts_module(self):
        assert DEEP_SYNTHESIS_PROMPT is not None
        assert "{reports}" in DEEP_SYNTHESIS_PROMPT
