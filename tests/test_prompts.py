"""Tests for tldr_scholar.prompts."""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path
from tldr_scholar.models import AudienceEnum, ToneEnum
from tldr_scholar.prompts import PromptBuilder


class TestPromptBuilder:
    def test_get_audience_instruction_expert(self):
        builder = PromptBuilder()
        instr = builder._get_audience_instruction(AudienceEnum.EXPERT)
        assert "precise technical terminology" in instr

    def test_build_system_prompt_persona(self, tmp_path):
        # Setup dummy persona
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        with open(persona_dir / "test.yaml", "w") as f:
            yaml.dump({
                "name": "test",
                "role": "academic",
                "tone": "sharp",
                "structure_pattern": "stitched"
            }, f)
            
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

    def test_build_system_prompt_bullet_pattern(self, tmp_path):
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        with open(persona_dir / "bullet.yaml", "w") as f:
            yaml.dump({
                "name": "bullet",
                "role": "data scientist",
                "tone": "concise",
                "structure_pattern": "bullet_points"
            }, f)
            
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
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        with open(persona_dir / "test.yaml", "w") as f:
            yaml.dump({"name": "test", "role": "r", "tone": "t", "structure_pattern": "s"}, f)
            
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
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        with open(persona_dir / "test.yaml", "w") as f:
            yaml.dump({"name": "test", "role": "r", "tone": "t", "structure_pattern": "stitched"}, f)
            
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
