"""Tests for tldr_scholar.cli."""
from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from tldr_scholar.cli import app
from tldr_scholar.config import GeminiConfig
from tldr_scholar.models import AudienceEnum, SummaryMetadata, SummaryResult, ToneEnum

runner = CliRunner()


def _mock_result(text="A summary.", hashtags=None, backend="extractive", input_type="text"):
    return SummaryResult(
        text=text,
        hashtags=hashtags or [],
        metadata=SummaryMetadata(
            source="test.txt",
            input_type=input_type,
            backend_used=backend,
            max_chars=500,
            focus="insights",
            char_count=len(text),
        ),
    )


class TestCliHelp:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "summarize" in result.output.lower() or "pdf" in result.output.lower()
        assert "--audience" in result.output
        assert "--tone" in result.output


class TestCliOutput:
    def test_text_format(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Some research text about machine learning transformers.")
        with patch("tldr_scholar.cli.summarize_file", return_value=_mock_result()):
            result = runner.invoke(app, [str(f)])
        assert result.exit_code == 0
        assert "A summary." in result.output


class TestCliAudienceToneFlags:
    def test_audience_flag_passed(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("text")
        with patch("tldr_scholar.cli.summarize_file") as mock_fn:
            mock_fn.return_value = _mock_result()
            runner.invoke(app, [str(f), "--audience", "layman"])
            assert mock_fn.call_args[1]["audience"] == AudienceEnum.LAYMAN

    def test_tone_flag_passed(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("text")
        with patch("tldr_scholar.cli.summarize_file") as mock_fn:
            mock_fn.return_value = _mock_result()
            runner.invoke(app, [str(f), "--tone", "analytical"])
            assert mock_fn.call_args[1]["tone"] == ToneEnum.ANALYTICAL

    def test_invalid_audience_rejected(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("text")
        result = runner.invoke(app, [str(f), "--audience", "invalid"])
        assert result.exit_code == 2

    def test_invalid_tone_rejected(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("text")
        result = runner.invoke(app, [str(f), "--tone", "invalid"])
        assert result.exit_code == 2
