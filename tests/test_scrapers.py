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
