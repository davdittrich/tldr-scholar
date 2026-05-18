# Implementation Plan: Social Feed Synthesis (v4)

## Goal
Add production-grade support for social media feed synthesis using parallel async ingestion, 12-month chronological sampling, and substantive link filtering.

## Mechanism
- **Async Scrapers**: Implement `MastodonScraper` and `BlueskyScraper` with `asyncio` and `httpx` to handle concurrency and rate limits.
- **Backoff & Jitter**: Integrated retry logic for 429 errors during feed and article ingestion.
- **Parallel Link Ingestion**: Use `asyncio.gather` with a semaphore (max 5) to fetch linked articles via `tldr_scholar.ingest`.
- **Substantive Domain Filter**: Whitelist-based filtering (`.edu`, `.gov`, known news/blog patterns) to skip GIF hosts, social media loops, and image trackers.
- **Atomic Bridge**: Map `SocialPost` to its ingested `SourceArticle` for the `decompose -> correlate` pipeline.

## Phase 1: Robust Scrapers (tldr-scholar-lwn.1)
- Create `tldr_scholar/scrapers.py` with `async` BaseScraper.
- Implement `SocialPost` model with `links: list[str]` field.
- Implement date-filtering (12 months) and boost/reply filtering.
- Implement exponential backoff for network requests.

## Phase 2: Concurrent Ingestion (tldr-scholar-lwn.3)
- Create `LinkIngester` class to manage article fetching.
- Implement domain whitelist logic to ignore noise URLs.
- Wire `asyncio.Semaphore(5)` to prevent IP bans.
- Cache results in `.cache/tldr-scholar/corpus/` by URL hash.

## Phase 3: CLI & Pipeline Wiring (tldr-scholar-lwn.2)
- Update `synthesize_style.py` main() to use `asyncio.run()`.
- Add `--skip-links` and `--concurrency` flags.
- Display progress bar during the link ingestion phase.
- Feed pairs `(SourceArticle, SocialPost)` into the atomic analytic engine.

## Phase 4: Verification
- `tests/test_scrapers.py`: Verify async retrieval and retry logic.
- Verify 12-month chronological window.
- Verify substantive filtering blocks noise URLs.

## Forbidden
- No synchronous serial fetching for links (must be async/parallel).
- No silent failures for rate limits (must backoff).

## Audit Strategy
- Mock 429 responses in tests to verify backoff logic.
- Verify domain whitelist correctly identifies substantive links.
- Profile execution time for 50-link feed (target < 60s).
