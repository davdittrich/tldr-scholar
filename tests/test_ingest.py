"""Tests for tldr_scholar.ingest."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tldr_scholar.ingest import (
    ingest,
    UnsupportedInputError,
    PasswordProtectedError,
    EmptyTextError,
)
from tldr_scholar.oa_fetch import OAResult


class TestIngestFile:
    def test_plain_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world. This is a test document.")
        text, input_type = ingest(str(f))
        assert input_type == "text"
        assert "Hello world" in text

    def test_markdown_file_strips_formatting(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\n**Bold** text with [link](http://example.com)")
        text, input_type = ingest(str(f))
        assert input_type == "markdown"
        assert "# Title" not in text  # heading stripped
        assert "**Bold**" not in text  # emphasis stripped
        assert "Bold" in text  # text preserved
        assert "link" in text  # link text preserved

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG")
        with pytest.raises(UnsupportedInputError, match="Unsupported file type"):
            ingest(str(f))

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ingest("/nonexistent/file.txt")

    def test_empty_text_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("   ")
        with pytest.raises(EmptyTextError):
            ingest(str(f))


class TestIngestUrl:
    def test_html_url(self):
        with patch("tldr_scholar.ingest._fetch_html", return_value="<html><body><p>Research findings.</p></body></html>"):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Research findings show X."
                text, input_type = ingest("https://example.com/article")
        assert input_type == "html"
        assert "Research findings" in text

    def test_unsupported_scheme(self):
        with pytest.raises(UnsupportedInputError, match="Unsupported URL scheme"):
            ingest("ftp://example.com/file")

    def test_empty_html_extraction(self):
        with patch("tldr_scholar.ingest._fetch_html", return_value="<html></html>"):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = None
                with pytest.raises(EmptyTextError):
                    ingest("https://example.com/empty")


class TestFetchHtml:
    def test_uses_curl_cffi_impersonation(self):
        """_fetch_html must call curl_cffi with impersonate='chrome124'."""
        with patch("tldr_scholar.ingest.curl_requests") as mock_curl:
            mock_session = MagicMock()
            mock_curl.Session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_curl.Session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value.text = "<html>text</html>"
            from tldr_scholar.ingest import _fetch_html
            _fetch_html("https://example.com")
            mock_curl.Session.assert_called_once_with(impersonate="chrome124")
            from tldr_scholar.ingest import _BROWSER_HEADERS
            mock_session.get.assert_called_once_with(
                "https://example.com",
                headers=_BROWSER_HEADERS,
                timeout=15,
                allow_redirects=True,
            )


class TestFileSizeLimit:
    def test_large_file_truncated(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 6_000_000)
        text, _ = ingest(str(f))
        assert len(text) <= 5_000_000


class TestFetchOAPdf:
    def test_raises_on_html_response(self):
        with patch("tldr_scholar.ingest.curl_requests") as mock_curl:
            mock_session = MagicMock()
            mock_curl.Session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_curl.Session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value.content = b"<html>Not a PDF</html>"
            from tldr_scholar.ingest import _fetch_oa_pdf
            with pytest.raises(EmptyTextError, match="non-PDF"):
                _fetch_oa_pdf("https://example.com/paper.pdf")

    def test_raises_on_network_error(self):
        with patch("tldr_scholar.ingest.curl_requests") as mock_curl:
            mock_session = MagicMock()
            mock_curl.Session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_curl.Session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.get.side_effect = Exception("net err")
            from tldr_scholar.ingest import _fetch_oa_pdf
            with pytest.raises(EmptyTextError, match="Failed to download"):
                _fetch_oa_pdf("https://example.com/paper.pdf")

    def test_returns_text_from_valid_pdf(self):
        with patch("tldr_scholar.ingest.curl_requests") as mock_curl:
            mock_session = MagicMock()
            mock_curl.Session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_curl.Session.return_value.__exit__ = MagicMock(return_value=False)
            mock_session.get.return_value.content = b"%PDF-1.4 fake"
            with patch("tldr_scholar.ingest.fitz") as mock_fitz:
                mock_fitz.open.return_value = MagicMock()
                with patch("tldr_scholar.ingest._pdf_doc_to_text", return_value="Paper text"):
                    from tldr_scholar.ingest import _fetch_oa_pdf
                    assert _fetch_oa_pdf("https://example.com/paper.pdf") == "Paper text"


class TestIngestUrlOAFallback:
    def test_oa_pdf_found_after_js_gate(self):
        js_html = '<head><meta name="citation_doi" content="10.1525/collabra.147309"/></head>' \
                  '<body>Enable JavaScript and cookies to continue</body>'
        with patch("tldr_scholar.ingest._fetch_html", return_value=js_html):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Enable JavaScript and cookies to continue"
                with patch("tldr_scholar.ingest.find_oa", return_value=OAResult(pdf_url="https://cdn.example.com/p.pdf")):
                    with patch("tldr_scholar.ingest._fetch_oa_pdf", return_value="Full paper text"):
                        text, input_type = ingest("https://online.ucpress.edu/collabra/article/12/1/147309/")
        assert input_type == "oa_pdf"
        assert text == "Full paper text"

    def test_abstract_fallback(self):
        js_html = '<meta name="citation_doi" content="10.1234/test"/>Enable JavaScript'
        with patch("tldr_scholar.ingest._fetch_html", return_value=js_html):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Enable JavaScript and cookies to continue"
                with patch("tldr_scholar.ingest.find_oa", return_value=OAResult(abstract="Emoji study abstract.")):
                    text, input_type = ingest("https://example.com/article")
        assert input_type == "abstract"
        assert "Emoji" in text

    def test_full_text_fallback(self):
        js_html = '<meta name="citation_doi" content="10.1234/test"/>Enable JavaScript'
        with patch("tldr_scholar.ingest._fetch_html", return_value=js_html):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Enable JavaScript and cookies to continue"
                with patch("tldr_scholar.ingest.find_oa", return_value=OAResult(full_text="Full paper body.")):
                    text, input_type = ingest("https://example.com/article")
        assert input_type == "oa_full_text"
        assert text == "Full paper body."

    def test_normal_page_skips_oa(self):
        with patch("tldr_scholar.ingest._fetch_html", return_value="<html><body>Real content.</body></html>"):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Real content."
                with patch("tldr_scholar.ingest.find_oa") as mock_oa:
                    ingest("https://example.com/article")
        mock_oa.assert_not_called()

    def test_raises_when_no_doi_and_js_gated(self):
        with patch("tldr_scholar.ingest._fetch_html", return_value="<html>Enable JavaScript and cookies to continue</html>"):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Enable JavaScript and cookies to continue"
                with pytest.raises(EmptyTextError, match="JavaScript"):
                    ingest("https://example.com/no-doi")

    def test_email_from_backend_config_forwarded_to_find_oa(self):
        """oa email in backend_config must be passed to find_oa."""
        js_html = '<meta name="citation_doi" content="10.1234/test"/>Enable JavaScript'
        with patch("tldr_scholar.ingest._fetch_html", return_value=js_html):
            with patch("tldr_scholar.ingest.trafilatura") as mock_traf:
                mock_traf.extract.return_value = "Enable JavaScript and cookies to continue"
                with patch("tldr_scholar.ingest.find_oa", return_value=None) as mock_oa:
                    with pytest.raises(EmptyTextError):
                        ingest("https://example.com/article",
                               backend_config={"oa": {"email": "user@example.com"}})
        mock_oa.assert_called_once_with("10.1234/test", email="user@example.com")
