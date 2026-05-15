"""Tests for tldr_scholar.backends."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from tldr_scholar.backends import get_backend, run_with_fallback
from tldr_scholar.backends.extractive import ExtractiveBackend
from tldr_scholar.backends.lemonade import LemonadeBackend, _MODEL_NAME_RE


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
        result = b.summarize(SAMPLE_TEXT, max_chars=200, focus="", hashtag_instruction="")
        assert result is not None
        assert len(result) > 0

    def test_focus_keyword_biasing(self):
        b = ExtractiveBackend()
        focused = b.summarize(SAMPLE_TEXT, max_chars=500, focus="transformer", hashtag_instruction="")
        unfocused = b.summarize(SAMPLE_TEXT, max_chars=500, focus="", hashtag_instruction="")
        # Both produce output; focus may reorder sentences
        assert focused is not None
        assert unfocused is not None

    def test_ignores_hashtag_instruction(self):
        b = ExtractiveBackend()
        result = b.summarize(SAMPLE_TEXT, max_chars=200, focus="", hashtag_instruction="Generate 5 hashtags")
        assert result is not None
        assert "#" not in result  # extractive doesn't generate hashtags


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
        result = b.summarize("text", 500, "insights", "")
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
        b.summarize("Document text here.", 500, "insights", "Generate 3 hashtags.")
        import json
        body = json.loads(captured["body"])
        messages = body["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
        # User message contains the document text
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Document text here." in user_msg["content"]

    @respx.mock
    def test_returns_none_on_failure(self):
        respx.post("http://127.0.0.1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="error")
        )
        b = LemonadeBackend({"model": "test-model", "host": "http://127.0.0.1:8000"})
        result = b.summarize("text", 500, "insights", "")
        assert result is None


class TestRunWithFallback:
    def test_auto_first_success_wins(self):
        with patch("tldr_scholar.backends.get_backend") as mock_get:
            mock_b = MagicMock()
            mock_b.summarize.return_value = "gemini result"
            mock_get.return_value = mock_b
            result, backend_used, _ = run_with_fallback("text", 500, "", "", "auto")
        assert result == "gemini result"
        assert backend_used == "gemini"

    def test_auto_fallback_on_failure(self):
        call_count = [0]
        def _get_backend(name, config=None):
            call_count[0] += 1
            mock_b = MagicMock()
            if name == "gemini":
                mock_b.summarize.return_value = None
            elif name == "lemonade":
                mock_b.summarize.return_value = "lemonade result"
            else:
                mock_b.summarize.return_value = None
            return mock_b

        with patch("tldr_scholar.backends.get_backend", side_effect=_get_backend):
            result, backend_used, _ = run_with_fallback("text", 500, "", "", "auto")
        assert result == "lemonade result"
        assert backend_used == "lemonade"

    def test_explicit_backend_no_fallback(self):
        with patch("tldr_scholar.backends.get_backend") as mock_get:
            mock_b = MagicMock()
            mock_b.summarize.return_value = None
            mock_get.return_value = mock_b
            result, _, _usage = run_with_fallback("text", 500, "", "", "gemini")
        assert result is None

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            run_with_fallback("text", 500, "", "", "nonexistent")

    def test_document_delimiters_in_prompt(self):
        """FR-16: single-prompt template contains <document> delimiters."""
        from tldr_scholar.prompts import SINGLE_PROMPT_TEMPLATE
        assert "<document>" in SINGLE_PROMPT_TEMPLATE
        assert "</document>" in SINGLE_PROMPT_TEMPLATE


class TestGeminiUsageThreading:
    def test_gemini_backend_stores_last_usage(self):
        from tldr_scholar.backends.gemini import GeminiBackend
        from gemini_acp.client import GeminiUsage

        backend = GeminiBackend({})
        mock_usage = GeminiUsage(tokens_used=500, cost_usd=0.001, cost_currency="USD")

        with patch("tldr_scholar.backends.gemini.summarize_via_gemini",
                   return_value=("summary text", mock_usage)):
            result = backend.summarize("text", 200, "focus", "", mode="general", sentence_count=3)

        assert result == "summary text"
        assert backend._last_usage is mock_usage

    def test_run_with_fallback_returns_3_tuple_with_usage(self):
        from gemini_acp.client import GeminiUsage

        mock_usage = GeminiUsage(tokens_used=300, cost_usd=0.0005, cost_currency="USD")
        mock_backend = MagicMock()
        mock_backend.summarize.return_value = "summary"
        mock_backend._last_usage = mock_usage

        with patch("tldr_scholar.backends.get_backend", return_value=mock_backend):
            response, name, usage = run_with_fallback(
                text="text", max_chars=200, focus="focus",
                hashtag_instruction="", backend="gemini", config=None,
            )

        assert response == "summary"
        assert usage is mock_usage

    def test_run_with_fallback_extractive_no_usage(self):
        response, name, usage = run_with_fallback(
            text=SAMPLE_TEXT,
            max_chars=200, focus="test", hashtag_instruction="",
            backend="extractive", config=None,
        )

        assert response is not None
        assert usage is None

    def test_summarize_sets_usage_metadata(self):
        from tldr_scholar import summarize
        from gemini_acp.client import GeminiUsage

        mock_usage = GeminiUsage(tokens_used=750, cost_usd=0.002, cost_currency="USD")

        with patch("tldr_scholar.run_with_fallback",
                   return_value=("summary", "gemini", mock_usage)):
            result = summarize(text="some text")

        assert result.metadata.tokens_used == 750
        assert result.metadata.cost_usd == 0.002
        assert result.metadata.cost_currency == "USD"

    def test_summarize_no_usage_metadata_is_none(self):
        from tldr_scholar import summarize

        with patch("tldr_scholar.run_with_fallback",
                   return_value=("summary", "extractive", None)):
            result = summarize(text="some text")

        assert result.metadata.tokens_used is None
        assert result.metadata.cost_usd is None

    def test_json_includes_usage_fields(self):
        from tldr_scholar.models import SummaryResult, SummaryMetadata
        import json

        result = SummaryResult(
            text="summary",
            hashtags=[],
            metadata=SummaryMetadata(
                backend_used="gemini", max_chars=200, focus="test",
                tokens_used=500, cost_usd=0.001, cost_currency="USD",
            ),
        )
        d = result.model_dump(mode="json")
        assert d["metadata"]["tokens_used"] == 500
        assert d["metadata"]["cost_usd"] == 0.001
        assert d["metadata"]["cost_currency"] == "USD"
