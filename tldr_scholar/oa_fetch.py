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
