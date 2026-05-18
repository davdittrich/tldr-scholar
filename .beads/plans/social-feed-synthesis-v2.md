# Implementation Plan: Social Feed Synthesis (v2)

## Goal
Add native support for synthesizing personas directly from social media feeds (Mastodon, Bluesky) using a robust 12-month sampling strategy.

## Mechanism
- **Scraper Factory**: New module `tldr_scholar/scrapers.py` with `BaseScraper` interface and Pydantic-validated `SocialPost` objects.
- **Resource-Guarded Sampling**: Chronological 12-month window back to `now - months`, capped at **MAX_POSTS = 200** to prevent context/token overflow.
- **Substantive Filtering**: 
  - Drop boosts/reposts and replies (`is_original` check).
  - Drop posts < 20 characters.
  - Strip HTML tags and normalize whitespace.
- **Error Handling**: Explicit failure reporting for feed access errors; no silent fallbacks to `ingest()`.

## Phase 1: Scraper Implementation (tldr-scholar-lwn.1)
- Define `SocialPost(text: str, timestamp: datetime, is_original: bool)` model.
- Implement `MastodonScraper`: 
  - Fetch `.rss` feed.
  - Parse `<pubDate>` using `email.utils.parsedate_to_datetime` (UTC).
  - Extract `<description>`, strip HTML.
- Implement `BlueskyScraper`: 
  - Fetch public profile page.
  - Extract posts from embedded JSON state (more stable than HTML parsing).
- Implement `ScraperFactory`: Regex-based dispatch with `unknown_url` exception.

## Phase 2: CLI & Pipeline Wiring (tldr-scholar-lwn.2)
- Update `tldr-scholar-synthesize-style` main():
  - Detect social URLs via `ScraperFactory`.
  - Add `--limit-months` (default: 12) and `--max-posts` (default: 200) flags.
  - Implement a `CorpusCache` for scraped text to enable fast re-runs.
- Integrate scraper output into the existing `decompose -> correlate -> synthesize` atomic loop.

## Phase 3: Verification
- `tests/test_scrapers.py`: Mock RSS and JSON responses.
- Verify 12-month boundary and `MAX_POSTS` cap logic.
- Verify substantive filters (length, original content).

## Forbidden
- No silent fallbacks when social scraping is explicitly requested.
- No mandatory API keys for public data.

## Audit Strategy
- `pytest`: All tests pass.
- Verify `SocialPost` validation for incoming feed items.
- Check loguru output for specific scraping error messages.
