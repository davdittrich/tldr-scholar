"""Social media feed scrapers with engagement tracking."""
from __future__ import annotations

import asyncio
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx
from loguru import logger
from pydantic import BaseModel, Field


def _backoff_delay(attempt: int, retry_after: Optional[float] = None) -> float:
    """Exponential backoff with jitter. Honors Retry-After if provided."""
    if retry_after is not None and retry_after > 0:
        return min(retry_after, 60.0) + random.uniform(0, 1.0)
    return min(1.0 * (2 ** attempt), 60.0) + random.uniform(0, 1.0)


class SocialPost(BaseModel):
    """Normalized social media post with engagement."""
    text: str
    timestamp: datetime
    is_original: bool = True
    links: list[str] = Field(default_factory=list)
    source_url: str
    engagement: int = 0 # sum of likes, boosts, etc.


@runtime_checkable
class BaseScraper(Protocol):
    """Interface for social media scrapers."""
    async def scrape(self, url: str, limit_months: int = 12, max_posts: int = 1000) -> list[SocialPost]:
        ...


class MastodonScraper:
    """Scraper for Mastodon feeds using public API."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def _get_with_retry(self, url: str, retries: int = 3) -> httpx.Response:
        for i in range(retries):
            try:
                resp = await self.client.get(url, timeout=15)
                if resp.status_code == 429:
                    retry_after_hdr = resp.headers.get("Retry-After")
                    retry_after = float(retry_after_hdr) if retry_after_hdr and retry_after_hdr.replace('.','',1).isdigit() else None
                    wait = _backoff_delay(i, retry_after=retry_after)
                    logger.warning(f"Rate limited (429). Retrying in {wait:.2f}s (attempt {i+1}/{retries})...")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if i == retries - 1:
                    raise
                await asyncio.sleep(_backoff_delay(i))
        raise Exception(f"Failed to fetch {url} after {retries} retries")

    def _strip_html(self, html: str) -> str:
        text = re.sub(r'<[^>]+>', '', html)
        return re.sub(r'\s+', ' ', text).strip()

    def _extract_links(self, html: str) -> list[str]:
        links = re.findall(r'href="([^"]+)"', html)
        substantive = []
        for link in links:
            if any(x in link for x in ['/tags/', '/users/', '/@', '/statuses/']):
                continue
            substantive.append(link)
        return substantive

    async def _get_account_id(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        instance = f"{parsed.scheme}://{parsed.netloc}"
        handle = parsed.path.split('/')[-1].lstrip('@')
        lookup_url = f"{instance}/api/v1/accounts/lookup?acct={handle}"
        try:
            resp = await self._get_with_retry(lookup_url)
            return resp.json().get("id")
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.error(f"Account lookup failed: {e}")
            return None

    async def scrape(self, url: str, limit_months: int = 12, max_posts: int = 1000) -> list[SocialPost]:
        """Fetch and parse Mastodon statuses via API (exhaustively)."""
        account_id = await self._get_account_id(url)
        if not account_id:
            return []

        parsed = urlparse(url)
        instance = f"{parsed.scheme}://{parsed.netloc}"
        base_api_url = f"{instance}/api/v1/accounts/{account_id}/statuses?limit=40&exclude_reblogs=true&exclude_replies=true"
        
        posts = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=limit_months * 30)
        max_id = None
        
        while len(posts) < max_posts:
            api_url = base_api_url
            if max_id:
                api_url += f"&max_id={max_id}"
                
            resp = await self._get_with_retry(api_url)
            data = resp.json()
            if not data:
                break
                
            for entry in data:
                ts = datetime.fromisoformat(entry["created_at"].replace('Z', '+00:00'))
                if ts < cutoff:
                    # Past window
                    return posts
                
                raw_html = entry["content"]
                text = self._strip_html(raw_html)
                if len(text) >= 20:
                    engagement = entry.get("reblogs_count", 0) + entry.get("favourites_count", 0) + entry.get("replies_count", 0)
                    posts.append(SocialPost(
                        text=text,
                        timestamp=ts,
                        is_original=True,
                        links=self._extract_links(raw_html),
                        source_url=entry["url"],
                        engagement=engagement
                    ))
                
                if len(posts) >= max_posts:
                    break
                
            max_id = data[-1]["id"]
            if len(data) < 40:
                break

        if posts:
            logger.info(f"Fetched {len(posts)} posts from Mastodon.")
        return posts


class BlueskyScraper:
    """Scraper for Bluesky feeds with engagement tracking."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def scrape(self, url: str, limit_months: int = 12, max_posts: int = 1000) -> list[SocialPost]:
        logger.info(f"Fetching Bluesky profile: {url}")
        parsed = urlparse(url)
        handle = parsed.path.split('/')[-1]
        if not handle:
            return []
            
        api_url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={handle}&filter=posts_and_author_threads&limit=100"
        
        posts = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=limit_months * 30)
        cursor = None

        while len(posts) < max_posts:
            fetch_url = api_url
            if cursor:
                fetch_url += f"&cursor={cursor}"
            
            try:
                resp = await self.client.get(fetch_url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                logger.error(f"Bluesky API fetch failed: {e}")
                break
                
            feed = data.get("feed", [])
            if not feed:
                break
                
            for entry in feed:
                post_data = entry.get("post", {})
                record = post_data.get("record", {})
                text = record.get("text", "")
                created_at = record.get("created_at")
                
                if not text or not created_at:
                    continue
                    
                ts = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                if ts < cutoff:
                    return posts
                    
                if len(text) < 20 or record.get("reply"):
                    continue

                links = []
                facets = record.get("facets", [])
                for facet in facets:
                    for feature in facet.get("features", []):
                        if feature.get("$type") == "app.bsky.richtext.facet#link":
                            links.append(feature.get("uri"))

                engagement = post_data.get("repostCount", 0) + post_data.get("likeCount", 0) + post_data.get("replyCount", 0)
                posts.append(SocialPost(
                    text=text,
                    timestamp=ts,
                    is_original=True,
                    links=links,
                    source_url=post_data.get("uri", url),
                    engagement=engagement
                ))
                
                if len(posts) >= max_posts:
                    break
            
            cursor = data.get("cursor")
            if not cursor:
                break
            
        return posts


class ScraperFactory:
    """Factory to get the correct scraper for a URL."""

    @staticmethod
    def get_scraper(url: str, client: httpx.AsyncClient) -> Optional[BaseScraper]:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if "bsky.app" in domain:
            return BlueskyScraper(client)
        if "/@" in parsed.path:
            return MastodonScraper(client)
        return None
