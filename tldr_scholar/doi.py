"""DOI extraction from URLs and HTML meta tags."""
from __future__ import annotations
import re

_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>#?&]+')

# Matches <meta ... citation_doi ... > in either attribute order
_META_DOI_RE = re.compile(
    r'<meta\b[^>]*\bcitation_doi\b[^>]*\bcontent=["\']([^"\']+)["\'][^>]*/?>',
    re.IGNORECASE,
)
# Also matches reversed order: content=... name=citation_doi
_META_DOI_RE_REV = re.compile(
    r'<meta\b[^>]*\bcontent=["\']([^"\']+)["\'][^>]*\bcitation_doi\b[^>]*/?>',
    re.IGNORECASE,
)


def extract_doi(url: str, html: str = "") -> str | None:
    """Return DOI from URL path or HTML <meta name="citation_doi">. None if not found."""
    if m := _DOI_RE.search(url):
        return m.group(0).rstrip(".,;)")
    if html:
        for pattern in (_META_DOI_RE, _META_DOI_RE_REV):
            if m := pattern.search(html):
                return m.group(1).strip()
    return None
