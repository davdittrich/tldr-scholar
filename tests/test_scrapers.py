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


def test_backoff_exponential_growth_and_jitter():
    """Verify _backoff_delay grows ~2^attempt, honors Retry-After, caps at 60s."""
    from tldr_scholar.scrapers import _backoff_delay

    # Capped: attempt=5 -> min(32, 60) + jitter[0,1] -> [32, 33]
    capped = _backoff_delay(5)
    assert 32.0 <= capped <= 33.0

    # Attempt 0: base 1.0 + jitter[0,1] -> [1, 2]
    a0 = _backoff_delay(0)
    assert 1.0 <= a0 <= 2.0

    # Attempt 2: base 4.0 + jitter[0,1] -> [4, 5]; > attempt 0 on average
    a2_samples = [_backoff_delay(2) for _ in range(5)]
    assert all(4.0 <= v <= 5.0 for v in a2_samples)

    # Jitter introduces variance across calls
    samples = [_backoff_delay(0) for _ in range(20)]
    assert len(set(samples)) > 1, "jitter should vary delay"

    # Retry-After is honored: 10 + jitter[0,1] -> [10, 11]
    ra = _backoff_delay(0, retry_after=10.0)
    assert 10.0 <= ra <= 11.0

    # Retry-After also capped at 60
    capped_ra = _backoff_delay(0, retry_after=300.0)
    assert 60.0 <= capped_ra <= 61.0


@pytest.mark.asyncio
async def test_429_retry_then_success(respx_mock):
    """Verify _get_with_retry retries on 429 and eventually succeeds."""
    from tldr_scholar.scrapers import MastodonScraper

    route = respx_mock.get("https://example.test/api/v1/something").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    async with httpx.AsyncClient() as client:
        s = MastodonScraper(client)
        resp = await s._get_with_retry("https://example.test/api/v1/something", retries=3)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_async_gather_preserves_order_in_process_posts():
    """Verify process_posts emits SourceArticles in input post order even with concurrent fetch."""
    from tldr_scholar.scrapers import SocialPost
    from tldr_scholar.ingestion_engine import LinkIngester
    from datetime import datetime, timezone
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ingester = LinkIngester(cache_dir=Path(tmp), concurrency=3)

        # Patch fetch_article: return predictable body per URL with variable delay
        # so completion order differs from submission order.
        async def fake_fetch(url):
            import asyncio
            # First URL waits longest -> would complete last under naive ordering
            delay = {"https://a.edu/x": 0.03, "https://b.edu/y": 0.01, "https://c.edu/z": 0.02}.get(url, 0)
            await asyncio.sleep(delay)
            return f"BODY:{url}"

        ingester.fetch_article = fake_fetch  # type: ignore[method-assign]

        now = datetime.now(timezone.utc)
        posts = [
            SocialPost(text="p0", timestamp=now, source_url="https://x.test/0",
                       links=["https://a.edu/x"], engagement=0),
            SocialPost(text="p1", timestamp=now, source_url="https://x.test/1",
                       links=["https://b.edu/y"], engagement=0),
            SocialPost(text="p2", timestamp=now, source_url="https://x.test/2",
                       links=["https://c.edu/z"], engagement=0),
        ]
        results = await ingester.process_posts(posts)

    assert len(results) == 3
    # Order MUST match input post order (gather preserves index order)
    assert results[0].url == "https://a.edu/x"
    assert results[1].url == "https://b.edu/y"
    assert results[2].url == "https://c.edu/z"
    assert results[0].body == "BODY:https://a.edu/x"
    assert results[1].body == "BODY:https://b.edu/y"
    assert results[2].body == "BODY:https://c.edu/z"
    assert results[0].post is posts[0]
    assert results[1].post is posts[1]
    assert results[2].post is posts[2]


def test_domain_whitelist_filters_non_substantive():
    """Verify is_substantive accepts whitelist domains/patterns and rejects others."""
    from tldr_scholar.ingestion_engine import LinkIngester
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ingester = LinkIngester(cache_dir=Path(tmp))

        # Rejected: social loops
        assert ingester.is_substantive("https://mastodon.social/@user") is False
        assert ingester.is_substantive("https://bsky.app/profile/foo") is False
        assert ingester.is_substantive("https://twitter.com/user/status/1") is False
        assert ingester.is_substantive("https://t.co/abc") is False
        assert ingester.is_substantive("https://fediscience.org/@dav") is False

        # Rejected: media/trackers
        assert ingester.is_substantive("https://youtube.com/watch?v=abc") is False
        assert ingester.is_substantive("https://imgur.com/gallery/x") is False
        assert ingester.is_substantive("https://giphy.com/gif/x") is False

        # Rejected: unknown generic domain
        assert ingester.is_substantive("https://random-site.example.com/post") is False

        # Accepted: substantive TLDs
        assert ingester.is_substantive("https://example.edu/paper") is True
        assert ingester.is_substantive("https://nih.gov/report") is True
        assert ingester.is_substantive("https://wikipedia.org/article") is True

        # Accepted: substantive patterns in netloc or path
        assert ingester.is_substantive("https://news.ycombinator.com/item") is True
        assert ingester.is_substantive("https://medium.com/blog/post") is True
        assert ingester.is_substantive("https://site.com/journal/2024") is True
        assert ingester.is_substantive("https://site.com/article/123") is True
        assert ingester.is_substantive("https://site.com/paper/abc") is True
