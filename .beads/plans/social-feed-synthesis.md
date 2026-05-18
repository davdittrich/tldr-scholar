<!-- SUPERSEDED-BY: social-feed-synthesis-v4.md (2026-05-18) -->
# Implementation Plan: Social Feed Synthesis

## Goal
Add native support for synthesizing personas directly from social media feeds (Mastodon, Bluesky) using an exhaustive 12-month sampling strategy.

## Mechanism
- **Scraper Factory**: New module `tldr_scholar/scrapers.py` to handle platform-specific extraction.
- **12-Month Sampling**: Fetch posts chronologically, covering the last 12 months (or up to platform limits).
- **Substantive Filtering**: Filter out short replies, boosts/reposts, and non-text media to ensure high-quality corpus.
- **Atomic Integration**: Feed the scraped corpus directly into the `decompose -> correlate -> synthesize` pipeline.

## Phase 1: Scraper Implementation (tldr-scholar-lwn.1)
- Implement `MastodonScraper`: Use `https://<instance>/@<user>.rss` or public HTML parsing for simple extraction without API keys.
- Implement `BlueskyScraper`: Use public web view or `atproto` unauthenticated fetch.
- Ensure chronological sorting and date filtering (12-month window).

## Phase 2: CLI & Pipeline Wiring (tldr-scholar-lwn.2)
- Update `tldr-scholar-synthesize-style` to detect social URLs.
- Integrate scraper into the main loop: `URL -> Scraped Corpus -> Atomic Pipeline`.
- Add `--limit-months` flag (default: 12).

## Phase 3: Verification
- Test with `https://fediscience.org/@davdittrich`.
- Verify the generated `davd.yaml` reflects the user's substantive stance.

## Forbidden
- No mandatory API keys for public feed reading (prefer public RSS/HTML where possible).
- No processing of boosts/reposts (focus on original content only).

## Audit Strategy
- Mock scraper responses in `tests/test_scrapers.py`.
- Verify 12-month window boundary logic.
- Run synthesis on a live feed and check for `agenda` extraction.
