import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from tldr_scholar.corpus_cache import CorpusCache
from tldr_scholar.scrapers import SocialPost


def _make_post(text: str = "x") -> SocialPost:
    return SocialPost(
        text=text,
        timestamp=datetime.now(timezone.utc),
        source_url="https://example.com/post/1",
        links=[],
        engagement=0,
    )


def test_corpus_cache_miss_on_empty():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(cache_dir=Path(tmp))
        assert cache.get("https://example.com", 12) is None


def test_corpus_cache_put_then_get_hits():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(cache_dir=Path(tmp))
        posts = [_make_post("hello")]
        cache.put("https://example.com", 12, posts)
        out = cache.get("https://example.com", 12)
        assert out is not None
        assert len(out) == 1
        assert out[0].text == "hello"


def test_corpus_cache_ttl_expires():
    with tempfile.TemporaryDirectory() as tmp:
        cache = CorpusCache(cache_dir=Path(tmp), ttl_seconds=0)
        cache.put("https://example.com", 12, [_make_post()])
        # ttl=0 means anything older than 0 seconds is expired
        time.sleep(0.01)
        assert cache.get("https://example.com", 12) is None
