import pytest
import httpx
from tldr_scholar.scrapers import MastodonScraper, BlueskyScraper, ScraperFactory, UnknownURLError

@pytest.mark.asyncio
async def test_factory_dispatch():
    async with httpx.AsyncClient() as client:
        m = ScraperFactory.get_scraper("https://fediscience.org/@davdittrich", client)
        assert isinstance(m, MastodonScraper)

        b = ScraperFactory.get_scraper("https://bsky.app/profile/user.bsky.social", client)
        assert isinstance(b, BlueskyScraper)

        with pytest.raises(UnknownURLError):
            ScraperFactory.get_scraper("https://google.com", client)

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


def test_scrapers_conform_to_base_protocol():
    from tldr_scholar.scrapers import BaseScraper, MastodonScraper, BlueskyScraper
    m = MastodonScraper(client=None)  # type: ignore[arg-type]
    b = BlueskyScraper(client=None)  # type: ignore[arg-type]
    assert isinstance(m, BaseScraper)
    assert isinstance(b, BaseScraper)


def test_source_article_model_fields():
    from tldr_scholar.scrapers import SourceArticle, SocialPost
    from datetime import datetime, timezone
    post = SocialPost(
        text="x",
        timestamp=datetime.now(timezone.utc),
        source_url="https://example.com",
        links=[],
        engagement=0,
    )
    art = SourceArticle(
        url="https://example.com",
        body="hello",
        fetched_at=datetime.now(timezone.utc),
        post=post,
    )
    assert art.url == "https://example.com"
    assert art.body == "hello"
    assert art.post.text == "x"


@pytest.mark.asyncio
async def test_process_posts_returns_source_article_list():
    from tldr_scholar.scrapers import SourceArticle, SocialPost
    from tldr_scholar.ingestion_engine import LinkIngester
    from datetime import datetime, timezone
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ingester = LinkIngester(cache_dir=Path(tmp), concurrency=1)
        posts = [
            SocialPost(
                text="no link",
                timestamp=datetime.now(timezone.utc),
                source_url="https://example.com/post/1",
                links=[],
                engagement=0,
            ),
        ]
        results = await ingester.process_posts(posts)
        assert isinstance(results, list)
        assert len(results) == 1
        assert all(isinstance(r, SourceArticle) for r in results)
        assert results[0].body is None
        assert results[0].url == ""
        assert results[0].post is posts[0]
