import pytest
import httpx
from tldr_scholar.scrapers import MastodonScraper, BlueskyScraper, ScraperFactory

@pytest.mark.asyncio
async def test_factory_dispatch():
    async with httpx.AsyncClient() as client:
        m = ScraperFactory.get_scraper("https://fediscience.org/@davdittrich", client)
        assert isinstance(m, MastodonScraper)
        
        b = ScraperFactory.get_scraper("https://bsky.app/profile/user.bsky.social", client)
        assert isinstance(b, BlueskyScraper)
        
        none = ScraperFactory.get_scraper("https://google.com", client)
        assert none is None

def test_mastodon_html_strip():
    scraper = MastodonScraper(None)
    html = "<p>Hello <b>World</b></p>"
    assert scraper._strip_html(html) == "Hello World"

def test_is_substantive_rejects_unknown_domain():
    from tldr_scholar.ingestion_engine import LinkIngester
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        ingester = LinkIngester(cache_dir=Path(tmp))
        # Generic domain not in any whitelist (no edu/gov/org TLD, no news/blog/article/journal/paper substring)
        assert ingester.is_substantive("https://random-site.example.com/post/123") is False
        # Known substantive (depends on SUBSTANTIVE_TLDS/PATTERNS — likely .edu)
        assert ingester.is_substantive("https://example.edu/paper") is True
        # Social loop excluded
        assert ingester.is_substantive("https://mastodon.example/@user") is False
