"""Input handling: PDF, HTML (URL), Markdown, and plain text ingestion."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import fitz
import httpx
import trafilatura
from curl_cffi import requests as curl_requests
from loguru import logger

_MAX_INPUT_BYTES = 5_000_000  # 5 MB cap

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_JS_GATE_PATTERNS = [
    "enable javascript",
    "javascript is disabled",
    "please enable javascript",
    "you need to enable javascript",
    "javascript and cookies",
]


class UnsupportedInputError(Exception):
    """Raised when input type is not supported."""


class PasswordProtectedError(Exception):
    """Raised when PDF is password-protected."""


class EmptyTextError(Exception):
    """Raised when no text could be extracted."""


def ingest(source: str) -> tuple[str, str]:
    """Dispatch input by type. Returns (text, input_type).

    Raises UnsupportedInputError, PasswordProtectedError, EmptyTextError.
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _ingest_url(source)
    if parsed.scheme and parsed.scheme not in ("", "file"):
        raise UnsupportedInputError(f"Unsupported URL scheme: {parsed.scheme}")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _ingest_pdf(path)
        input_type = "pdf"
    elif ext == ".md":
        text = _ingest_markdown(path)
        input_type = "markdown"
    elif ext == ".txt":
        text = _ingest_text(path)
        input_type = "text"
    else:
        raise UnsupportedInputError(f"Unsupported file type: {ext}")

    if not text or not text.strip():
        raise EmptyTextError(f"No text could be extracted from {source}")
    return text, input_type


def _pdf_doc_to_text(doc, max_pages: int = 20) -> str:
    """Convert an open fitz.Document to text. Lazy-imports pymupdf4llm."""
    import pymupdf4llm
    pages = list(range(min(max_pages, len(doc))))
    try:
        text = pymupdf4llm.to_markdown(doc, pages=pages)
    except Exception:
        return ""
    return text.strip() if text else ""


def _ingest_pdf(path: Path) -> str:
    """Extract text from PDF (first 20 pages). Detects password-protected PDFs."""
    pdf_bytes = path.read_bytes()
    if len(pdf_bytes) > _MAX_INPUT_BYTES:
        logger.warning(f"PDF {path} exceeds {_MAX_INPUT_BYTES} bytes, truncating")
        pdf_bytes = pdf_bytes[:_MAX_INPUT_BYTES]

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            raise PasswordProtectedError(f"PDF is password-protected: {path}")
        raise

    if doc.is_encrypted:
        raise PasswordProtectedError(f"PDF is password-protected: {path}")

    return _pdf_doc_to_text(doc)


def _fetch_html(url: str) -> str:
    """Fetch URL HTML with Chrome TLS impersonation via curl-cffi."""
    with curl_requests.Session(impersonate="chrome124") as session:
        resp = session.get(url, headers=_BROWSER_HEADERS, timeout=15, allow_redirects=True)
        return resp.text


def _fetch_oa_pdf(url: str) -> str:
    """Download and parse a PDF from an OA URL. Raises EmptyTextError on failure."""
    try:
        with curl_requests.Session(impersonate="chrome124") as session:
            resp = session.get(url, headers=_BROWSER_HEADERS, timeout=30, allow_redirects=True)
            pdf_bytes = resp.content
    except Exception as e:
        raise EmptyTextError(f"Failed to download OA PDF from {url}: {e}")

    if len(pdf_bytes) > _MAX_INPUT_BYTES:
        logger.warning(f"OA PDF from {url} exceeds {_MAX_INPUT_BYTES} bytes, truncating")
        pdf_bytes = pdf_bytes[:_MAX_INPUT_BYTES]

    if not pdf_bytes.lstrip()[:5].startswith(b'%PDF-'):
        raise EmptyTextError(f"OA URL returned non-PDF content: {url}")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise EmptyTextError(f"Failed to parse OA PDF from {url}")

    text = _pdf_doc_to_text(doc)
    if not text:
        raise EmptyTextError(f"No text extracted from OA PDF: {url}")
    return text


def _ingest_url(url: str) -> tuple[str, str]:
    """Fetch URL, detect type, extract text. Returns (text, input_type)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsupportedInputError(f"Only http/https URLs supported, got: {parsed.scheme}")

    # Detect content type
    content_type = ""
    try:
        with httpx.Client(timeout=10, follow_redirects=True, headers=_BROWSER_HEADERS) as client:
            head = client.head(url)
            content_type = head.headers.get("content-type", "")
    except Exception:
        pass

    is_pdf = "pdf" in content_type.lower() or parsed.path.lower().endswith(".pdf")

    if is_pdf:
        # Download PDF bytes
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=30, headers=_BROWSER_HEADERS)
            pdf_bytes = resp.content
            if len(pdf_bytes) > _MAX_INPUT_BYTES:
                logger.warning(f"PDF from {url} exceeds {_MAX_INPUT_BYTES} bytes, truncating")
                pdf_bytes = pdf_bytes[:_MAX_INPUT_BYTES]
        except Exception as e:
            raise EmptyTextError(f"Failed to download PDF from {url}: {e}")

        # Detect if the server returned HTML instead of a PDF (JS gate, auth wall)
        if not pdf_bytes.lstrip()[:5].startswith(b'%PDF-'):
            html = pdf_bytes.decode("utf-8", errors="replace")
            if any(pat in html.lower() for pat in _JS_GATE_PATTERNS):
                raise EmptyTextError(
                    f"Publisher requires JavaScript/authentication — direct PDF access is blocked: {url}"
                )
            raise EmptyTextError(
                f"URL returned non-PDF content (likely an auth wall or redirect): {url}"
            )

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception:
            raise EmptyTextError(f"Failed to parse PDF from {url}")

        text = _pdf_doc_to_text(doc)
        if not text:
            raise EmptyTextError(f"No text could be extracted from {url}")
        return text, "pdf"
    else:
        # HTML — fetch with Chrome TLS impersonation
        try:
            html = _fetch_html(url)
        except Exception as e:
            raise EmptyTextError(f"Failed to fetch {url}: {e}")

    
        text = trafilatura.extract(html, output_format="txt",
                                   include_comments=False, include_tables=True)
        if not text or not text.strip():
            raise EmptyTextError(f"No text could be extracted from {url}")
        text = text.strip()
        lower = text.lower()
        if any(pat in lower for pat in _JS_GATE_PATTERNS):
            raise EmptyTextError(
                f"Page requires JavaScript to render — try the PDF URL directly: {url}"
            )
        return text, "html"


def _ingest_markdown(path: Path) -> str:
    """Read markdown file, strip formatting."""
    raw = _read_with_limit(path)
    # Strip markdown formatting: headings, links, emphasis, images
    text = re.sub(r'^#+\s+', '', raw, flags=re.MULTILINE)  # headings
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # images
    text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)  # links → text
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)  # emphasis
    text = re.sub(r'`([^`]+)`', r'\1', text)  # inline code
    text = re.sub(r'```[\s\S]*?```', '', text)  # code blocks
    return text.strip()


def _ingest_text(path: Path) -> str:
    """Read plain text file."""
    return _read_with_limit(path).strip()


def _read_with_limit(path: Path) -> str:
    """Read file with 5MB limit. Warns on truncation."""
    size = path.stat().st_size
    if size > _MAX_INPUT_BYTES:
        logger.warning(f"{path} is {size} bytes, truncating to {_MAX_INPUT_BYTES}")
    with open(path, "r", errors="replace") as f:
        return f.read(_MAX_INPUT_BYTES)
