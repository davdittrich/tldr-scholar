# Active Plan
<!-- approved: pending -->
<!-- gate-iterations: 2 (pragmatic-path) -->
<!-- user-approved: pending -->
<!-- status: drafted -->
<!-- epics: tldr-scholar-qe4, tldr-scholar-zaf, tldr-scholar-4lg -->

# Epic Plan v3 — Land v4 social-feed + Critical review remediation + v2 residuals

## Three-epic scope (26 atomic children open)

### Epic tldr-scholar-qe4 — Complete social-feed-synthesis v4 (P2, 11 open)
qe4.1 commit baseline (root, blocks rest); qe4.11 declare dev deps (pytest-asyncio + rich); qe4.12 SUPERSEDED-BY annotation on v1/v2 plans; qe4.2 ScraperFactory.get_scraper raise UnknownURLError; qe4.3 exponential backoff; qe4.7 SourceArticle bridge (blocks qe4.4); qe4.4 all substantive links; qe4.5 progress bar; qe4.6 --skip-links flag; qe4.8 CorpusCache; qe4.9 test expansion (deps on qe4.2/3/4/5/6/7/8/11). qe4.10 CLOSED (--concurrency already exists in worktree).

### Epic tldr-scholar-zaf — Critical review remediation (P2, 13 open)
Blocking (5): zaf.1 is_substantive whitelist; zaf.2 BaseScraper inheritance; zaf.3 LLM index validation; zaf.5 narrow scrapers exceptions; zaf.7 narrow ingestion exceptions.
Required (4): zaf.4 MD5 usedforsecurity=False; zaf.6 drop unused imports; zaf.8 scope httpx client; zaf.9 hoist corpus init (UnboundLocalError, rescoped from "orphan").
Suggestions (4): zaf.10 skip no-op asyncio.sleep tasks; zaf.11 generator return for process_posts; zaf.12 explicit httpx import in tests; zaf.13 log skipped pairs in decomposition loop.

### Epic tldr-scholar-4lg — v2 residuals cleanup (P3, 2 open)
4lg.1 print→logger (depends qe4.1); tldr-scholar-2k7 unused imports os/Any in personas.py.

## Iter-2 gate fixes applied (Path B: pragmatic)

| Finding | Action |
|---|---|
| qe4.10 redundant (--concurrency already exists) | CLOSED with reason |
| 4lg.1 missing qe4.1 dep | dep added |
| Ticket count drift | recounted: 26 open |
| zaf.9 misdiagnosis (orphan var) | rescoped to UnboundLocalError fix |
| qe4.2 title cosmetic (from_url vs get_scraper) | title corrected |
| 5 critical-review Suggestions unticketed | re-spawned reviewer, filed zaf.10-13 (S1 print overlap → folded into 4lg.1) |

## Iter-2 findings ACCEPTED as implementation parameters (not scope creep)

| Finding | Rationale |
|---|---|
| qe4.3 backoff params (base=1, cap=60, jitter_max=1) | Standard HTTP retry defaults (RFC 6585, AWS SDK convention) |
| qe4.7 SourceArticle field set | Derived from existing worktree usage; ticket guards "do NOT add fields beyond what consumers use" |
| qe4.8 TTL=1h | Reasonable default; can be tuned in follow-up |
| qe4.11 `rich` runtime dep | Required to honor v4 spec progress bar; ticket scope acknowledges deviation |
| qe4.5 rich vs tqdm choice | Single-library choice committed at planning to avoid runtime dispatch complexity |
| qe4.12 SUPERSEDED-BY housekeeping | Closes audit loop; prevents repeat misclassification |

## DoD (epic level)
- [ ] All 26 child tickets closed
- [ ] Full pytest green
- [ ] `tldr-scholar-synthesize-style https://fediscience.org/@davdittrich` runs end-to-end
- [ ] All commits reference parent epic ID
- [ ] `bd preflight` passes
- [ ] `git status` clean post-push
