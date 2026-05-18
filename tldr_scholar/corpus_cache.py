"""Disk cache for scraped social-feed corpora.

Per-stage caching with cascade invalidation (WU-5b, nwl.6).

Stage hierarchy (ordered):
    scrape -> cluster -> correlate -> aggregate

``invalidate(stage)`` wipes that stage and all downstream stages.
Cache keys include: stage + url + embedding_model + corpus_hash + flag_fingerprint.

Backwards-compatible scrape shim
---------------------------------
The old two-argument ``get(url, months)`` / ``put(url, months, posts)`` API is
preserved via overloaded method signatures.  Callers that have not yet migrated
to the new stage-keyed API continue to work transparently.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Literal, Optional, overload

from loguru import logger

from tldr_scholar.scrapers import SocialPost

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES: tuple[str, ...] = ("scrape", "cluster", "correlate", "aggregate")

STAGE_CASCADE: dict[str, list[str]] = {
    "scrape":    ["scrape", "cluster", "correlate", "aggregate"],
    "cluster":   ["cluster", "correlate", "aggregate"],
    "correlate": ["correlate", "aggregate"],
    "aggregate": ["aggregate"],
    "all":       list(STAGES),
}

DEFAULT_TTL_SECONDS = 3600  # 1 hour — applied to the legacy scrape shim only


class CorpusCache:
    """Disk-backed cache for pipeline stage data, keyed by stage + content-derived hash.

    Supports two APIs:

    New (per-stage) API
    ~~~~~~~~~~~~~~~~~~~
    .. code-block:: python

        cache.get(stage, key_dict)     # -> Any | None
        cache.put(stage, key_dict, value)
        cache.invalidate(stage_or_all)

    Legacy scrape-cache shim (URL + months)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. code-block:: python

        cache.get(url: str, months: int)          # -> list[SocialPost] | None
        cache.put(url: str, months: int, posts)
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        # Legacy kwarg kept for backwards compat with old callers using cache_dir=
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        # Accept both root= (new) and cache_dir= (old) keyword
        resolved_root = root or cache_dir or Path.home() / ".cache" / "tldr-scholar" / "stages"
        self.root = Path(resolved_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

        # Legacy scrape cache directory lives alongside new stage dirs for isolation
        self._scrape_legacy_dir = self.root / "_scrape_legacy"
        self._scrape_legacy_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-stage API
    # ------------------------------------------------------------------

    def _key_hash(self, stage: str, key_dict: dict[str, Any]) -> str:
        normalized = json.dumps({"stage": stage, **key_dict}, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _path(self, stage: str, key_hash: str) -> Path:
        return self.root / stage / f"{key_hash}.json"

    def _stage_get(self, stage: str, key_dict: dict[str, Any]) -> Any | None:
        p = self._path(stage, self._key_hash(stage, key_dict))
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def _stage_put(self, stage: str, key_dict: dict[str, Any], value: Any) -> None:
        p = self._path(stage, self._key_hash(stage, key_dict))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, default=str))

    # ------------------------------------------------------------------
    # Legacy scrape shim (URL + months keyed, TTL-expiring)
    # ------------------------------------------------------------------

    def _legacy_key(self, url: str, months: int) -> str:
        return hashlib.sha256(f"{url}|{months}".encode(), usedforsecurity=False).hexdigest()

    def _legacy_path(self, key: str) -> Path:
        return self._scrape_legacy_dir / f"{key}.json"

    def _legacy_get(self, url: str, months: int) -> Optional[list[SocialPost]]:
        path = self._legacy_path(self._legacy_key(url, months))
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
            logger.warning(f"CorpusCache legacy parse error: {e}")
            return None

    def _legacy_put(self, url: str, months: int, posts: list[SocialPost]) -> None:
        path = self._legacy_path(self._legacy_key(url, months))
        raw = [p.model_dump(mode="json") for p in posts]
        path.write_text(json.dumps(raw))
        logger.debug(f"CorpusCache PUT ({len(posts)} posts): {url}")

    # ------------------------------------------------------------------
    # Unified public API — dispatch on argument types
    # ------------------------------------------------------------------

    def get(self, stage_or_url: str, key_dict_or_months: "dict[str, Any] | int") -> Any | None:
        """Retrieve a cached value.

        New API:  ``get(stage: str, key_dict: dict) -> Any | None``
        Legacy:   ``get(url: str, months: int)      -> list[SocialPost] | None``
        """
        if isinstance(key_dict_or_months, int):
            # Legacy path: (url, months)
            return self._legacy_get(stage_or_url, key_dict_or_months)
        # New per-stage path: (stage, key_dict)
        return self._stage_get(stage_or_url, key_dict_or_months)

    def put(
        self,
        stage_or_url: str,
        key_dict_or_months: "dict[str, Any] | int",
        value: Any = None,
    ) -> None:
        """Store a value in the cache.

        New API:  ``put(stage: str, key_dict: dict, value: Any)``
        Legacy:   ``put(url: str, months: int, posts: list[SocialPost])``
        """
        if isinstance(key_dict_or_months, int):
            # Legacy path: (url, months, posts)
            self._legacy_put(stage_or_url, key_dict_or_months, value)
            return
        # New per-stage path: (stage, key_dict, value)
        self._stage_put(stage_or_url, key_dict_or_months, value)

    def invalidate(
        self,
        stage_or_all: Literal["scrape", "cluster", "correlate", "aggregate", "all"],
    ) -> None:
        """Wipe ``stage_or_all`` and all downstream stage directories.

        Idempotent: does not raise if directories do not exist.
        """
        for s in STAGE_CASCADE[stage_or_all]:
            stage_dir = self.root / s
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
                logger.debug(f"CorpusCache invalidated stage dir: {stage_dir}")
