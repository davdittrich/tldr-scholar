"""Concurrent article ingestion and filtering."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from tldr_scholar.ingest import ingest
from tldr_scholar.scrapers import SocialPost, SourceArticle

SUBSTANTIVE_TLDS = {".edu", ".gov", ".org"}
SUBSTANTIVE_PATTERNS = ["news", "blog", "article", "journal", "paper"]

class LinkIngester:
    """Manages parallel ingestion of linked articles from social posts."""

    def __init__(self, cache_dir: Optional[Path] = None, concurrency: int = 5):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "tldr-scholar" / "corpus"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Article cache: {self.cache_dir}")
        self.semaphore = asyncio.Semaphore(concurrency)

    def is_substantive(self, url: str) -> bool:
        """Filter for URLs likely to contain substantive article text."""
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Skip social media loops
        if any(x in netloc for x in ["t.co", "bsky.app", "mastodon", "fediscience", "twitter.com"]):
            return False
            
        # Check TLDs
        if any(netloc.endswith(tld) for tld in SUBSTANTIVE_TLDS):
            return True
            
        # Check patterns
        if any(p in netloc or p in path for p in SUBSTANTIVE_PATTERNS):
            return True
            
        # Default: lean permissive for academic domains but skip known trackers/media
        if any(x in netloc for x in ["youtube.com", "imgur.com", "giphy.com"]):
            return False

        logger.debug(f"is_substantive rejected (no whitelist match): {url}")
        return False

    async def fetch_article(self, url: str) -> Optional[str]:
        """Fetch and cache article content."""
        if not self.is_substantive(url):
            return None
            
        url_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()
        cache_path = self.cache_dir / f"{url_hash}.txt"
        
        if cache_path.exists():
            return cache_path.read_text()
            
        async with self.semaphore:
            logger.debug(f"Ingesting linked article: {url}")
            try:
                # Wrap ingest (sync) in thread for async safety
                text, _ = await asyncio.to_thread(ingest, url)
                if text:
                    cache_path.write_text(text)
                    return text
            except (httpx.HTTPError, ValueError, IOError) as e:
                logger.warning(f"Failed to ingest {url}: {e}")
                
        return None

    async def process_posts(self, posts: list[SocialPost]) -> list[SourceArticle]:
        """Parallel process ALL substantive links across posts (one SourceArticle per link).

        For posts with NO substantive links, emit a single SourceArticle(body=None, url="") to preserve
        a per-post record. For posts with multiple substantive links, emit one SourceArticle per link.
        """
        tasks: list = []
        owner_idx: list[int] = []
        owner_link: list[str] = []
        post_link_count: list[int] = [0] * len(posts)

        for idx, post in enumerate(posts):
            links = [l for l in post.links if self.is_substantive(l)]
            post_link_count[idx] = len(links)
            for link in links:
                tasks.append(self.fetch_article(link))
                owner_idx.append(idx)
                owner_link.append(link)

        fetched = await asyncio.gather(*tasks) if tasks else []
        success = len([b for b in fetched if b])
        logger.info(f"Ingestion complete: {success}/{len(fetched)} successful link fetches across {len(posts)} posts")
        now = datetime.now(timezone.utc)

        articles: list[SourceArticle] = []
        # For posts with no substantive link, still emit a SourceArticle so the consumer
        # has a 1:1 fallback. For posts with links, emit one SourceArticle per link.
        post_has_emitted = [False] * len(posts)
        for owner, link, body in zip(owner_idx, owner_link, fetched):
            articles.append(SourceArticle(url=link, body=body, fetched_at=now, post=posts[owner]))
            post_has_emitted[owner] = True
        for idx, emitted in enumerate(post_has_emitted):
            if not emitted:
                articles.append(SourceArticle(url="", body=None, fetched_at=now, post=posts[idx]))
        return articles
