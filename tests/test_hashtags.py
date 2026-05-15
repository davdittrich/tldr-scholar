"""Tests for tldr_scholar.hashtags."""
from __future__ import annotations

from tldr_scholar.hashtags import (
    build_hashtag_instruction,
    parse_hashtags_from_response,
    generate_hashtags_tfidf,
    format_pascal_case,
)


class TestBuildHashtagInstruction:
    def test_zero_returns_empty(self):
        assert build_hashtag_instruction(0) == ""

    def test_positive_returns_instruction(self):
        result = build_hashtag_instruction(5, style="pascal")
        assert "5 hashtags" in result
        assert "PascalCase" in result
        assert "#MachineLearning" in result


class TestFormatPascalCase:
    def test_basic(self):
        assert format_pascal_case("machine learning") == "#MachineLearning"
        assert format_pascal_case("artificial intelligence") == "#ArtificialIntelligence"

    def test_acronyms(self):
        assert format_pascal_case("ai research") == "#AIResearch"
        assert format_pascal_case("nlp model") == "#NLPModel"

    def test_numbers(self):
        assert format_pascal_case("web3 tech") == "#Web3Tech"

    def test_existing_caps(self):
        assert format_pascal_case("DeepSeek model") == "#DeepSeekModel"


class TestParseHashtagsFromResponse:
    def test_response_with_hashtags(self):
        response = "This is a summary.\n\n#AI #MachineLearning"
        summary, tags = parse_hashtags_from_response(response)
        assert summary == "This is a summary."
        assert "#AI" in tags
        assert "#MachineLearning" in tags


class TestGenerateHashtagsTfidf:
    def test_pascal_style(self):
        text = "Machine learning is a subset of artificial intelligence."
        tags = generate_hashtags_tfidf(text, 2, style="pascal")
        assert "#MachineLearning" in tags or "#ArtificialIntelligence" in tags
        assert all(t.startswith("#") for t in tags)
        # Verify no underscores in PascalCase tags
        for t in tags:
            assert "_" not in t
