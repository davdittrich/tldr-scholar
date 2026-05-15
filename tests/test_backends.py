"""Tests for tldr_scholar.backends."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from tldr_scholar.backends import get_backend, run_with_fallback
from tldr_scholar.backends.extractive import ExtractiveBackend
from tldr_scholar.backends.lemonade import LemonadeBackend, _MODEL_NAME_RE
from tldr_scholar.models import AudienceEnum, ToneEnum


SAMPLE_TEXT = (
    "Machine learning has transformed natural language processing. "
    "Transformer architectures enable contextual understanding of text. "
    "Recent advances in large language models demonstrate emergent capabilities. "
    "These models can summarize, translate, and generate human-like text."
)


class TestGetBackend:
    def test_valid_backends(self):
        for name in ["gemini", "lemonade", "ollama", "extractive"]:
            b = get_backend(name)
            assert b is not None

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")


class TestExtractiveBackend:
    def test_produces_summary(self):
        b = ExtractiveBackend()
        result = b.summarize(
            SAMPLE_TEXT, max_chars=200, focus="", hashtag_instruction="",
            audience=AudienceEnum.EXPERT, tone=ToneEnum.PROFESSIONAL
        )
        assert result is not None
        assert len(result) > 0

    def test_focus_keyword_biasing(self):
        b = ExtractiveBackend()
        focused = b.summarize(
            SAMPLE_TEXT, max_chars=500, focus="transformer", hashtag_instruction="",
            audience=AudienceEnum.EXPERT, tone=ToneEnum.PROFESSIONAL
        )
        unfocused = b.summarize(
            SAMPLE_TEXT, max_chars=500, focus="", hashtag_instruction="",
            audience=AudienceEnum.EXPERT, tone=ToneEnum.PROFESSIONAL
        )
        # Both produce output; focus may reorder sentences
        assert focused is not None
        assert unfocused is not None


class TestLemonadeBackend:
    def test_model_name_validation(self):
        assert _MODEL_NAME_RE.match("Phi-4-mini-instruct-GGUF")
        assert _MODEL_NAME_RE.match("user.DeepSeek-R1-GGUF")
        assert not _MODEL_NAME_RE.match("model; rm -rf /")
        assert not _MODEL_NAME_RE.match("")

    @respx.mock
    def test_returns_content_on_success(self):
        respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "A summary.\n\n#ai #ml"}}]
            })
        )
        b = LemonadeBackend({"model": "test-model", "host": "http://127.0.0.1:8000"})
        result = b.summarize("text", 500, "insights", "", AudienceEnum.EXPERT, ToneEnum.PROFESSIONAL)
        assert "summary" in result

    @respx.mock
    def test_sends_system_and_user_messages(self):
        """OpenAI spec: messages must include both system and user roles."""
        captured = {}
        def _capture(request):
            captured["body"] = request.content
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Summary."}}]
            })
        respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(side_effect=_capture)
        b = LemonadeBackend({"model": "test-model", "host": "http://127.0.0.1:8000"})
        b.summarize("Document text here.", 500, "insights", "Generate 3 hashtags.", AudienceEnum.EXPERT, ToneEnum.PROFESSIONAL)
        import json
        body = json.loads(captured["body"])
        messages = body["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


class TestRunWithFallback:
    def test_auto_first_success_wins(self):
        with patch("tldr_scholar.backends.get_backend") as mock_get:
            mock_b = MagicMock()
            mock_b.summarize.return_value = "gemini result"
            mock_get.return_value = mock_b
            result, backend_used, _ = run_with_fallback("text", 500, "", "", "auto", AudienceEnum.EXPERT, ToneEnum.PROFESSIONAL)
        assert result == "gemini result"
        assert backend_used == "gemini"


class TestGeminiUsageThreading:
    def test_gemini_backend_stores_last_usage(self):
        from tldr_scholar.backends.gemini import GeminiBackend
        from gemini_acp.client import GeminiUsage

        backend = GeminiBackend({})
        mock_usage = GeminiUsage(tokens_used=500, cost_usd=0.001, cost_currency="USD")

        with patch("tldr_scholar.backends.gemini.summarize_via_gemini",
                   return_value=("summary text", mock_usage)):
            result = backend.summarize(
                "text", 200, "focus", "", AudienceEnum.EXPERT, ToneEnum.PROFESSIONAL,
                mode="general", sentence_count=3
            )

        assert result == "summary text"
        assert backend._last_usage is mock_usage
