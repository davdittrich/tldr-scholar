"""Tests for tldr_scholar.doi."""
from __future__ import annotations
import pytest
from tldr_scholar.doi import extract_doi


class TestExtractDoi:
    def test_springer_url(self):
        assert extract_doi("https://link.springer.com/article/10.1007/s00265-021-03123-4") \
            == "10.1007/s00265-021-03123-4"

    def test_wiley_url(self):
        assert extract_doi("https://onlinelibrary.wiley.com/doi/10.1002/anie.202101234") \
            == "10.1002/anie.202101234"

    def test_plos_url(self):
        assert extract_doi("https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123456") \
            == "10.1371/journal.pone.0123456"

    def test_ucpress_opaque_url_returns_none(self):
        assert extract_doi("https://online.ucpress.edu/collabra/article/12/1/147309/217078/Emojis") \
            is None

    def test_ucpress_meta_tag_name_before_content(self):
        """Standard attribute order: name=... content=..."""
        html = '<meta name="citation_doi" content="10.1525/collabra.147309"/>'
        assert extract_doi(
            "https://online.ucpress.edu/collabra/article/12/1/147309/217078/Emojis",
            html=html,
        ) == "10.1525/collabra.147309"

    def test_ucpress_meta_tag_content_before_name(self):
        """Reversed attribute order: content=... name=..."""
        html = '<meta content="10.1525/collabra.147309" name="citation_doi"/>'
        assert extract_doi(
            "https://online.ucpress.edu/collabra/article/12/1/147309/217078/Emojis",
            html=html,
        ) == "10.1525/collabra.147309"

    def test_nature_opaque_slug_with_meta(self):
        html = '<meta name="citation_doi" content="10.1038/s41586-024-07528-6">'
        assert extract_doi("https://www.nature.com/articles/s41586-024-07528-6", html=html) \
            == "10.1038/s41586-024-07528-6"

    def test_no_doi_anywhere_returns_none(self):
        assert extract_doi("https://example.com/page", html="<html></html>") is None

    def test_trailing_punctuation_stripped(self):
        assert extract_doi("https://example.com/paper/10.1234/test.abc.") == "10.1234/test.abc"

    def test_meta_tag_case_insensitive(self):
        html = '<meta name="Citation_DOI" content="10.9999/example">'
        assert extract_doi("https://example.com", html=html) == "10.9999/example"
