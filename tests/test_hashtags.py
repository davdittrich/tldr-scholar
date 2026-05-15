"""Tests for tldr_scholar.hashtags."""
from __future__ import annotations

from tldr_scholar.hashtags import (
    build_hashtag_instruction,
    parse_hashtags_from_response,
    generate_hashtags_tfidf,
)


class TestBuildHashtagInstruction:
    def test_zero_returns_empty(self):
        assert build_hashtag_instruction(0) == ""

    def test_negative_returns_empty(self):
        assert build_hashtag_instruction(-1) == ""

    def test_positive_returns_instruction(self):
        result = build_hashtag_instruction(5)
        assert "5 hashtags" in result
        assert len(result) > 0
        assert "#machine_learning" in result or "#climate_change" in result


class TestParseHashtagsFromResponse:
    def test_response_with_hashtags(self):
        response = "This is a summary of the paper.\n\n#ai #machine_learning #nlp"
        summary, tags = parse_hashtags_from_response(response)
        assert summary == "This is a summary of the paper."
        assert "#ai" in tags
        assert "#machine_learning" in tags
        assert "#nlp" in tags

    def test_response_without_hashtags(self):
        response = "Just a plain summary with no tags."
        summary, tags = parse_hashtags_from_response(response)
        assert summary == response
        assert tags == []


class TestGenerateHashtagsTfidf:
    def test_produces_hashtags(self):
        text = (
            "Machine learning transforms natural language processing. "
            "Transformer architectures enable contextual understanding. "
            "Large language models demonstrate emergent capabilities."
        )
        tags = generate_hashtags_tfidf(text, 3)
        assert len(tags) == 3
        assert all(t.startswith("#") for t in tags)

    def test_captures_bigrams(self):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "Recent machine learning advances are significant. "
            "We study machine learning models."
        )
        tags = generate_hashtags_tfidf(text, 3)
        # Should capture 'machine_learning' as a single tag
        assert "#machine_learning" in tags

    def test_filters_stopword_bigrams(self):
        text = (
            "The study of the paper and of the results. "
            "The results of the study show that of the data is clean."
        )
        tags = generate_hashtags_tfidf(text, 5)
        # Should NOT capture 'of_the' or 'the_study' if they are too generic
        assert "#of_the" not in tags

    def test_hashtags_are_lowercase(self):
        text = "Machine Learning and Natural Language Processing are important."
        tags = generate_hashtags_tfidf(text, 3)
        for tag in tags:
            assert tag == tag.lower()
