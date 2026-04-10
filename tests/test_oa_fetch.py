"""Tests for tldr_scholar.oa_fetch."""
from __future__ import annotations

import httpx
import respx

from tldr_scholar.oa_fetch import (
    OAResult,
    query_unpaywall,
    query_openalex,
    query_semantic_scholar,
    find_oa,
)


class TestUnpaywall:
    @respx.mock
    def test_returns_pdf_url_when_oa_available(self):
        doi = "10.1525/collabra.147309"
        respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "best_oa_location": {
                        "url_for_pdf": "https://cdn.example.com/paper.pdf"
                    }
                },
            )
        )
        result = query_unpaywall(doi, email="test@test.com")
        assert result is not None
        assert result.pdf_url == "https://cdn.example.com/paper.pdf"

    @respx.mock
    def test_returns_none_when_no_oa_location(self):
        doi = "10.9999/closed"
        respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
            return_value=httpx.Response(200, json={"best_oa_location": None})
        )
        assert query_unpaywall(doi) is None

    @respx.mock
    def test_returns_none_on_404(self):
        doi = "10.9999/missing"
        respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
            return_value=httpx.Response(404)
        )
        assert query_unpaywall(doi) is None

    def test_returns_none_on_network_error(self):
        with respx.mock:
            respx.get("https://api.unpaywall.org/v2/10.9/err").mock(
                side_effect=httpx.ConnectError("refused")
            )
            assert query_unpaywall("10.9/err") is None

    @respx.mock
    def test_email_passed_as_query_param(self):
        doi = "10.1234/test"
        route = respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
            return_value=httpx.Response(200, json={"best_oa_location": None})
        )
        query_unpaywall(doi, email="me@example.com")
        assert "me" in str(route.calls[0].request.url)


class TestOpenAlex:
    @respx.mock
    def test_returns_pdf_url(self):
        doi = "10.1234/oa"
        respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(
            return_value=httpx.Response(200, json={
                "best_oa_location": {"pdf_url": "https://oa.example.com/paper.pdf"},
                "abstract_inverted_index": None,
            })
        )
        assert query_openalex(doi).pdf_url == "https://oa.example.com/paper.pdf"

    @respx.mock
    def test_reconstructs_abstract_from_inverted_index(self):
        doi = "10.1234/abs"
        respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(
            return_value=httpx.Response(200, json={
                "best_oa_location": None,
                "abstract_inverted_index": {"Hello": [0], "world": [1]},
            })
        )
        result = query_openalex(doi)
        assert "Hello" in result.abstract and "world" in result.abstract

    @respx.mock
    def test_returns_none_on_404(self):
        respx.get("https://api.openalex.org/works/doi:10.9/x").mock(return_value=httpx.Response(404))
        assert query_openalex("10.9/x") is None


class TestSemanticScholar:
    @respx.mock
    def test_returns_pdf_and_abstract(self):
        doi = "10.1234/s2"
        respx.get(f"https://api.semanticscholar.org/graph/v1/paper/{doi}").mock(
            return_value=httpx.Response(200, json={
                "abstract": "Studies emoji at work.",
                "openAccessPdf": {"url": "https://s2.example.com/paper.pdf"},
            })
        )
        result = query_semantic_scholar(doi)
        assert result.pdf_url == "https://s2.example.com/paper.pdf"
        assert result.abstract == "Studies emoji at work."

    @respx.mock
    def test_abstract_only_when_no_oa_pdf(self):
        doi = "10.1234/abs"
        respx.get(f"https://api.semanticscholar.org/graph/v1/paper/{doi}").mock(
            return_value=httpx.Response(200, json={"abstract": "Abstract only.", "openAccessPdf": None})
        )
        result = query_semantic_scholar(doi)
        assert result.pdf_url is None
        assert result.abstract == "Abstract only."

    @respx.mock
    def test_returns_none_on_404(self):
        respx.get("https://api.semanticscholar.org/graph/v1/paper/10.9/x").mock(
            return_value=httpx.Response(404)
        )
        assert query_semantic_scholar("10.9/x") is None


class TestFindOA:
    @respx.mock
    def test_unpaywall_wins_first(self):
        doi = "10.1525/collabra.147309"
        respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
            return_value=httpx.Response(200, json={
                "best_oa_location": {"url_for_pdf": "https://cdn.example.com/paper.pdf"}
            })
        )
        assert find_oa(doi).pdf_url == "https://cdn.example.com/paper.pdf"

    @respx.mock
    def test_falls_through_to_openalex_when_unpaywall_fails(self):
        doi = "10.1234/fallthrough"
        respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(return_value=httpx.Response(404))
        respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(
            return_value=httpx.Response(200, json={
                "best_oa_location": {"pdf_url": "https://oa.example.com/p.pdf"},
                "abstract_inverted_index": None,
            })
        )
        assert find_oa(doi).pdf_url == "https://oa.example.com/p.pdf"

    @respx.mock
    def test_returns_none_when_all_fail(self):
        doi = "10.9999/closed"
        respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(return_value=httpx.Response(404))
        respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(return_value=httpx.Response(404))
        respx.get(f"https://api.semanticscholar.org/graph/v1/paper/{doi}").mock(
            return_value=httpx.Response(404)
        )
        assert find_oa(doi) is None

    @respx.mock
    def test_email_forwarded_to_unpaywall(self):
        """find_oa must forward email to query_unpaywall."""
        doi = "10.1234/email-test"
        route = respx.get(f"https://api.unpaywall.org/v2/{doi}").mock(
            return_value=httpx.Response(200, json={
                "best_oa_location": {"url_for_pdf": "https://cdn.example.com/p.pdf"}
            })
        )
        find_oa(doi, email="configured@example.com")
        assert "configured" in str(route.calls[0].request.url)
