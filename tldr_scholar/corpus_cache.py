"""Disk cache for scraped social-feed corpora."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from tldr_scholar.scrapers import SocialPost

DEFAULT_TTL_SECONDS = 3600  # 1 hour


class CorpusCache:
    """Disk-backed cache for lists of SocialPost keyed by feed URL + month window."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "tldr-scholar" / "feeds"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _key(self, url: str, months: int) -> str:
        return hashlib.sha256(f"{url}|{months}".encode(), usedforsecurity=False).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, url: str, months: int) -> Optional[list[SocialPost]]:
        path = self._path(self._key(url, months))
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            logger.debug(f"CorpusCache MISS (expired, age={age:.0f}s > ttl={self.ttl_seconds}s): {url}")
            return None
        try:
            raw = json.loads(path.read_text())
            posts = [SocialPost.model_validate(item) for item in raw]
            logger.debug(f"CorpusCache HIT ({len(posts)} posts): {url}")
            return posts
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"CorpusCache invalidate (parse error): {e}")
            return None

    def put(self, url: str, months: int, posts: list[SocialPost]) -> None:
        path = self._path(self._key(url, months))
        raw = [p.model_dump(mode="json") for p in posts]
        path.write_text(json.dumps(raw))
        logger.debug(f"CorpusCache PUT ({len(posts)} posts): {url}")
