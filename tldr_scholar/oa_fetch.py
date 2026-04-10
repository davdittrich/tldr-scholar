"""Open-access API clients for finding freely available paper content."""
from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger

_TIMEOUT = 10


@dataclass
class OAResult:
    """Result from an open-access query."""

    pdf_url: str | None = None
    abstract: str | None = None
    full_text: str | None = None


def query_unpaywall(doi: str, email: str = "tldr-scholar@localhost") -> OAResult | None:
    """Query Unpaywall for an OA PDF URL. Returns None on any failure.

    Args:
        doi: Digital Object Identifier (e.g., "10.1525/collabra.147309")
        email: Email to pass to Unpaywall API (required by their terms)

    Returns:
        OAResult with pdf_url if open access available, None otherwise.
    """
    try:
        resp = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        loc = resp.json().get("best_oa_location") or {}
        pdf_url = loc.get("url_for_pdf") or None
        return OAResult(pdf_url=pdf_url) if pdf_url else None
    except Exception as e:
        logger.debug(f"Unpaywall failed for {doi}: {e}")
        return None


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract from OpenAlex inverted word index."""
    if not inverted_index:
        return None
    words = sorted(
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    )
    return " ".join(w for _, w in words)


def query_openalex(doi: str) -> OAResult | None:
    """Query OpenAlex for OA PDF and abstract. Returns None on any failure.

    Args:
        doi: Digital Object Identifier

    Returns:
        OAResult with pdf_url and/or abstract if available, None otherwise.
    """
    try:
        resp = httpx.get(
            f"https://api.openalex.org/works/doi:{doi}",
            params={"mailto": "tldr-scholar@localhost"},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        pdf_url = ((data.get("best_oa_location") or {}).get("pdf_url")) or None
        abstract = _reconstruct_abstract(data.get("abstract_inverted_index"))
        return OAResult(pdf_url=pdf_url, abstract=abstract) if (pdf_url or abstract) else None
    except Exception as e:
        logger.debug(f"OpenAlex failed for {doi}: {e}")
        return None


def query_semantic_scholar(doi: str) -> OAResult | None:
    """Query Semantic Scholar for OA PDF and abstract. Returns None on any failure.

    Args:
        doi: Digital Object Identifier

    Returns:
        OAResult with pdf_url and/or abstract if available, None otherwise.
    """
    try:
        resp = httpx.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{doi}",
            params={"fields": "abstract,openAccessPdf"},
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        pdf_url = ((data.get("openAccessPdf") or {}).get("url")) or None
        abstract = data.get("abstract") or None
        return OAResult(pdf_url=pdf_url, abstract=abstract) if (pdf_url or abstract) else None
    except Exception as e:
        logger.debug(f"Semantic Scholar failed for {doi}: {e}")
        return None


def find_oa(doi: str, email: str = "") -> OAResult | None:
    """Try OA APIs in priority order. Returns first result with content.

    Tries Unpaywall → OpenAlex → Semantic Scholar. Stops at first result
    with pdf_url, abstract, or full_text.

    Args:
        doi: Digital Object Identifier
        email: Email to pass to Unpaywall (optional; defaults to localhost)

    Returns:
        OAResult with pdf_url/abstract/full_text, or None if all fail.
    """
    _email = email or "tldr-scholar@localhost"
    for fn, kw in [
        (query_unpaywall, {"email": _email}),
        (query_openalex, {}),
        (query_semantic_scholar, {}),
    ]:
        result = fn(doi, **kw)
        if result and (result.pdf_url or result.full_text or result.abstract):
            logger.debug(f"OA found via {fn.__name__} for doi:{doi}")
            return result
    return None
