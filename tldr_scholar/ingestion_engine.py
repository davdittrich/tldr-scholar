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
        """Parallel process all links in a list of posts."""
        tasks: list = []
        task_post_idx: list[int] = []
        post_links: list[Optional[str]] = [None] * len(posts)
        for idx, post in enumerate(posts):
            # For now, we only take the FIRST substantive link per post
            link = next((l for l in post.links if self.is_substantive(l)), None)
            post_links[idx] = link
            if link:
                tasks.append(self.fetch_article(link))
                task_post_idx.append(idx)

        fetched = await asyncio.gather(*tasks) if tasks else []
        bodies: list[Optional[str]] = [None] * len(posts)
        for i, body in zip(task_post_idx, fetched):
            bodies[i] = body
        success = len([b for b in bodies if b])
        logger.info(f"Ingestion complete: {success} success, {len(bodies)-success} skipped/failed")
        now = datetime.now(timezone.utc)
        return [
            SourceArticle(
                url=post_links[i] or "",
                body=bodies[i],
                fetched_at=now,
                post=posts[i],
            )
            for i in range(len(posts))
        ]
