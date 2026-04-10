"""Tests for tldr_scholar.oa_fetch."""
from __future__ import annotations

import httpx
import respx

from tldr_scholar.oa_fetch import OAResult, query_unpaywall


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
