# Persona Neutral-Baseline Delta — Design

**Date:** 2026-05-18
**Status:** Revised v3 — design-review-gate rounds 1 + 2 incorporated, all reviewers APPROVED
**Author:** persona-creator brainstorm session
**Related epic:** tldr-scholar-0pb (Improve Generated Text Quality)

## Motivation

User feedback on existing tldr-scholar personas: generated summaries are off-topic and under-personalized. Current pipeline collapses persona-emphasis patterns into a single global priority list, so topic-specific framing (e.g., persona suppresses stats in econ posts but shares them in tech) averages into noise. This delta replaces classification + correlation with a multi-baseline topic-aware pipeline so generation can match the article's topic to the persona's per-topic emphasis pattern.

## Current pipeline gaps

1. **Loses framing/emphasis signal.** Atomic claims discard the abstraction level a neutral summary preserves (e.g., a paper claim "X causes Y, N=50" decomposes to flat claims; the SUMMARY would have bundled N=50 as a qualifier — persona dropping the qualifier shows up as framing, not claim omission).
2. **Per-post deltas only, no per-topic aggregation.** Patterns aggregate globally into one flat agenda. Per-topic emphasis averages into noise.
3. **LLM-driven topic classification.** Gemini `CLASSIFICATION_PROMPT` is expensive, non-deterministic, provides no embeddings for downstream similarity at generation time.
4. **Generation doesn't mirror selective-emphasis pattern.** Applies a global priority list. Cannot match the article's topic to the persona's per-topic emphasis pattern.

## Goal

Replace classification + correlation stages with a local-NLP + multi-baseline pipeline that:
- captures both per-claim AND framing-level deltas
- aggregates per-topic
- emits per-topic centroids so generation can blend topic profiles by article-to-centroid similarity

## Non-goals

- Multi-author persona blending
- Persona persistence beyond YAML files
- Real-time persona refinement during generation
- Replacing Gemini with a local LLM for summarization

## Design overview (Approach B — pipeline replacement)

```
posts → embed → cluster → TopicProfile per cluster
              ↓
        per-topic correlation (3 baselines × posts)
              ↓
        per-topic aggregation (1 Gemini call/topic)
              ↓
        global synthesis
              ↓
        Persona v2 (top-level + topics dict + topic_centroids)
```

### Module structure

- `tldr_scholar/topic_cluster.py` — `cluster_posts(posts) -> (labels: list[str], centroids: dict[str, list[float]])`. Module-level singleton for sentence-transformers model (lazy-loaded once per process); embedding function is a free function, NOT a method on PromptBuilder.
- `tldr_scholar/source_baseline.py` — `class SourceBaselines: claims, extractive_summary, abstractive_summary`.
- `tldr_scholar/correlator.py` — `correlate_against_baselines(post, baselines) -> list[DeltaRecord]`.
- `tldr_scholar/synthesize_style.py` — orchestrates the new pipeline. Drops `CLASSIFICATION_PROMPT`. Adds `NEUTRAL_SUMMARY_PROMPT`. Replaces `correlate_post_to_source` invocation with `correlator.correlate_against_baselines`.
- `tldr_scholar/personas.py` — adds `TopicProfile` Pydantic model. `Persona` gains `topics: dict[str, TopicProfile]` and `embedding_model: str`. Loader is **v2-only**: no v1 shape detection, no migration logic. Old v1 files are deleted in phase 1.
- `tldr_scholar/prompts.py` — `PromptBuilder.build_system_prompt` for persona path detects non-empty `topics`, calls the embedding singleton from `topic_cluster`, computes softmax-weighted blend over top-K topics, builds blended `revelation_priorities` / `suppression_rules` text. PromptBuilder itself remains pure-string; it imports the embed function rather than owning the model.
- `tldr_scholar/scrape_filter.py` — **NEW**: `is_likely_injection(text: str) -> bool`. Pipeline:
  1. Normalize input: NFKC unicode normalization + strip zero-width / control characters (U+200B, U+200C, U+200D, U+FEFF, control range).
  2. Regex scan the normalized form for prompt-injection markers (see fixture `tests/fixtures/injection_patterns.txt`): "ignore previous instructions", role tokens (`<|im_start|>`, `<system>`, `<|assistant|>`), "you are now", "disregard the above", instruction-override phrasings, base-prompt leak attempts.
  3. Match → drop pre-clustering. Emit warn envelope with `code: "injection_filter_match"` and `drops[].source` only. **Never echo the matched content in any envelope field.**

### New runtime dependencies

- `sentence-transformers>=2.2`
- `hdbscan>=0.8`
- `bertopic>=0.16` — used for c-TF-IDF topic labeling (avoids hand-rolling)

`sumy` already present.

### Persona schema v2

```python
class TopicProfile(BaseModel):
    label: str                              # human-readable, auto-labeled by bertopic c-TF-IDF top-3 terms
    centroid: list[float]                   # 384-d for all-MiniLM-L6-v2 (normalized)
    sample_size: int                        # number of posts in cluster
    posts: list[str]                        # FULL clustered post text (preserved for explainability + re-aggregation)
    revelation_priorities: list[str]
    suppression_rules: list[str]
    substantive_anchors: list[str]
    rhetorical_strategy: str
    confidence: dict[str, float]

class DeltaRecord(BaseModel):
    baseline_type: Literal["claims", "extractive", "abstractive"]
    statements: list[str]
    status_per_statement: list[Literal["shared", "suppressed", "distorted"]]
    intent: str | None

class Persona(BaseModel):
    name: str
    embedding_model: str                    # mandatory in v2; mismatch at gen time → fail loud
    status: Literal["complete", "incomplete"] = "complete"
    incomplete_stages: list[str] = Field(default_factory=list)
    agenda: str
    worldview: str
    pivot_logic: str
    identifiable_nuances: list[str]
    attribute_confidence: dict[str, int] = Field(default_factory=dict)
    topics: dict[str, TopicProfile]         # mandatory; min 1 topic (fallback "_global")
```

`DeltaRecord.status_per_statement` uses `Literal[str]` (not Pydantic Enum) to avoid the strict-matching gotcha under Gemini JSON-mode responses.

### Topic clustering details

- Model: `sentence-transformers/all-MiniLM-L6-v2` (~50MB, CPU, 384-d output)
- HDBSCAN(min_cluster_size=5, min_samples=3, metric='cosine')
- Noise points (label = -1) → bucketed under `"_unclustered"` topic
- If HDBSCAN yields 0 real clusters → fallback single `"_global"` topic containing all posts
- Centroid = mean of normalized embeddings within cluster, re-normalized
- Auto-labeling: bertopic c-TF-IDF top-3 terms per cluster joined `+`. Example: `"economics+labor+wage"`. Deterministic, no LLM call.

### Baseline generation

Per matched (post, source) pair:

| Baseline | Source | Compute | Deterministic |
|----------|--------|---------|---------------|
| atomic claims | Gemini `DECOMPOSITION_PROMPT` (existing) | 1 LLM call | no |
| extractive summary | sumy `LexRankSummarizer`, 5 sentences | local CPU | yes |
| abstractive summary | Gemini `NEUTRAL_SUMMARY_PROMPT` (new), 3 sentences | 1 LLM call | no |

`NEUTRAL_SUMMARY_PROMPT`:
```
Produce a neutral 3-sentence summary of the following text. Report findings only.
No opinion, no agenda, no editorial framing. Preserve numerical specifics
(N, p-values, percentages) when present.

<untrusted_content>
{source_text}
</untrusted_content>

Return ONLY the 3-sentence summary as plain text. No headers, no bullets.
```

All prompts that take scraped/external content wrap it in `<untrusted_content>...</untrusted_content>` delimiters with an explicit instruction to treat the block as data, not instructions. This applies to `NEUTRAL_SUMMARY_PROMPT`, `DECOMPOSITION_PROMPT`, `CORRELATION_PROMPT`, and any future prompt taking external text.

### Correlation

Per (post, baselines) triple:

- For claims baseline: re-use existing CORRELATION_PROMPT (with claims as statements input).
- For extractive: each sentence is one statement.
- For abstractive: each sentence is one statement.

Output: 3 DeltaRecord per (post, source).

### Per-topic aggregation

**Contract: 1 Gemini call per topic, full aggregation.** For each topic, collect all DeltaRecord across all posts in cluster. The single LLM call produces:

- `revelation_priorities`: top-N statements with `status="shared"` weighted by frequency
- `suppression_rules`: top-N with `status="suppressed"`
- `substantive_anchors`: top-N with `status="distorted"`
- `rhetorical_strategy`: 1-sentence summary
- `confidence`: per-attribute percentage of baselines agreeing

(Earlier draft offered a local-frequency alternative; rejected for contract clarity.)

### Global aggregation

Run existing `DEEP_SYNTHESIS_PROMPT` over all DeltaRecord (across all topics) → populates Persona top-level `agenda` / `worldview` / `pivot_logic` / `identifiable_nuances`.

### Generation blending

In `prompts.py` `PromptBuilder.build_system_prompt` persona path:

1. Call `topic_cluster.embed_text(input_source)` (singleton model). Returns 384-d normalized vector.
2. For each topic: `cosine_sim = dot(input_emb, centroid)`.
3. Softmax with temperature τ=0.3 over similarities → weight per topic.
4. Drop topics with weight < 0.1.
5. Top-K=3 retained topics.
6. Build blended priority/suppression lists: weighted-frequency vote across retained TopicProfiles.
7. Inject into existing PERSONA_SYSTEM_PROMPT placeholders (no template change).

### Cost defaults

`--full-baselines` is **opt-in**. Default behavior uses claims baseline only (matches current pipeline cost ~200 LLM calls). When user invokes `--full-baselines`, abstractive + correlation across all 3 baselines runs (~755 LLM calls, ~3.8x).

Rationale: iterative authoring is the common case; full re-derivation is the rare case. Quality regen is explicit, not silent.

### Re-run semantics

`--reset=<stage>` flag (granular). Valid values: `cluster`, `correlate`, `aggregate`, `all`. Default: preserve all stages.

| Value | Wipes | Re-derives |
|-------|-------|-----------|
| (omitted) | nothing | nothing reused unless explicitly cached |
| `cluster` | cluster + correlate + aggregate (downstream invalidated) | clustering, then all downstream |
| `correlate` | correlate + aggregate | DeltaRecord per post, then aggregation |
| `aggregate` | aggregate only | per-topic LLM aggregation only (cheap re-aggregation, retains expensive baselines + DeltaRecord cache) |
| `all` | every cached stage | full pipeline from raw posts |

CorpusCache (existing infrastructure) is the persistence layer for cluster + correlate + aggregate stage outputs. Cache keys include `embedding_model` + corpus hash + relevant CLI flags so stale caches invalidate automatically across model/config changes.

### Error contract

**All failures emit JSON envelope on stderr.** Format:

```json
{"level":"error|warn","stage":"scrape|cluster|baseline|correlate|aggregate|generate","code":"<symbol>","message":"<human>","drops":[{"source":"<url|id>","reason":"<symbol>"}]}
```

**Content echo policy:** envelope MUST NOT include matched injection content, scraped post bodies, API keys, env vars, or absolute filesystem paths. The `message` field is for human-readable diagnosis only and is subject to the same rule. The `drops[].source` field carries source identifiers (URLs, post IDs) only — when a source URL contains query params or auth tokens (e.g., signed URLs), scrub query string before emit.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | unexpected internal error |
| 2 | v2 schema validation failure on persona load |
| 3 | embedding model mismatch at generation time |
| 4 | setup-time / unrecoverable I/O error (e.g., output file write fails) — persona may not be written |
| 5 | empty corpus after scrape + injection filter |

**Partial-write semantics for graceful degradation:** when per-stage LLM operations fail (e.g., per-topic aggregation, global synthesis), the aggregator returns a fallback result and the persona proceeds with `status: "incomplete"` and `incomplete_stages: list[str]` listing affected stages (e.g., `["aggregate_topic:partial"]` if per-topic aggregation had parse failures; `["aggregate_global:failed"]` if global synthesis failed). Loader accepts incomplete personas at generation time but emits warn envelope `persona_incomplete` on every gen call. User resumes by re-running with `--reset=<failed-stage>` to pick up from the failure point.

End-of-run summary (always emitted on stdout, even on exit 0) lists count of dropped sources/baselines with reasons.

### Failure modes

| Failure | Behavior |
|---------|----------|
| sentence-transformers model download fails | emit error envelope `model_download_failed`, exit 1 (no degraded mode — model is required) |
| HDBSCAN yields 0 clusters | single `"_global"` topic (logged warn envelope, exit 0) |
| sumy summary fails | drop extractive baseline for that source, continue with 2 baselines (warn envelope) |
| Gemini abstractive fails | drop abstractive baseline, continue with claims + extractive (warn envelope) |
| Gemini correlation fails | drop DeltaRecord (warn envelope, source listed in drops[]) |
| Per-topic or global aggregation fails | emit warn envelope, set `status=incomplete` in persona YAML with corresponding stage in `incomplete_stages` (e.g., `"aggregate_topic:partial"` or `"aggregate_global:failed"`), and exit 0 (graceful degradation) |
| All baselines fail for a source | drop source from corpus entirely (warn envelope, source listed) |
| Centroid model mismatch at gen | error envelope `embedding_model_mismatch`, exit 3 |
| Post matches injection-marker regex | drop post pre-clustering (warn envelope `injection_filter_match`) |
| v1-shape persona file at load | error envelope `unsupported_persona_schema`, exit 2 (no auto-migrate) |

Error messages use **persona_dir-relative paths**, never absolute. Never include API key, full source content, or home-directory layout in error strings.

### Compute budget

Per persona run with ~200 sampled posts, ~150 with linked sources:

| Stage | LLM calls | Local time |
|-------|-----------|-----------|
| Embed posts | 0 | ~0.5s |
| Cluster | 0 | <1s |
| Decompose (claims) | 150 | — |
| Abstractive summary (--full-baselines only) | 150 | — |
| Extractive summary | 0 | <2s |
| Correlate (--full-baselines: 3 × 150; default: 1 × 150) | 450 / 150 | — |
| Per-topic aggregate | ~5-10 (one per topic) | — |
| Global synthesis | 1 | — |
| **Total default** | **~165-170** | **~5s** |
| **Total --full-baselines** | **~755-760** | **~5s** |

Default matches current pipeline cost (~200 calls). `--full-baselines` is the 3.8x mode.

### Backward compatibility

**None.** v1 persona files (`davd.yaml`, `davdittrich.yaml`, `samples.yaml`) are deleted in phase 1. v2 schema removes obsolete top-level fields (`revelation_priorities`, `suppression_rules`, `substantive_anchors`, `rhetorical_strategy`) which now exclusively live in TopicProfile. Loader is v2-only: any file failing v2 validation → exit 2 with `unsupported_persona_schema`. No auto-migration, no degraded-mode fallback.

### Corpus sampling for training + eval

1. **Pull last-year window.** Scrape all posts (with linked sources) from the persona's feed published in the past 12 months.
2. **Topic-balanced training sample.** Cluster the full window using the same `cluster_posts` pipeline, then sample N_train=200 posts balanced across topics/domains (proportional with floor=10/topic where possible). This is the corpus passed into baseline + correlation + aggregation.
3. **Per-topic eval sets.** For each discovered topic:
   - Hold out N_judge=10 articles for LLM-as-judge pairwise comparison
   - Hold out N_manual=5 articles for manual side-by-side scoring
4. All four numbers (`window_months`, `n_train`, `n_judge_per_topic`, `n_manual_per_topic`) configurable at runtime via CLI flags with defaults above.

### Ship-gate evaluation protocol

Quality validation combines two methods on the held-out per-topic eval sets. **No v1 baseline comparison**: v1 is deleted in WU-1; ship-gate uses intrinsic quality measures only.

1. **Manual side-by-side scoring (gating).** Author reads `N_manual_per_topic × topic_count` v2 persona outputs. Scores 1-5 on: (a) topical relevance, (b) style fidelity. **Target: mean ≥ 4.0 on both, no individual score below 3.**

2. **LLM intrinsic rating (advisory).** Same articles run through v2. Gemini receives each output alone, rates 1-5 on: (a) topical fit, (b) voice coherence. **Target: mean ≥ 4.0.**

**Ship-gate rule:**
- Manual mean ≥4.0 on both dimensions AND LLM intrinsic mean ≥4.0 → **ship**
- Manual fails → **block** (regardless of LLM rating)
- Manual passes but LLM intrinsic <4.0 → **triage**: author re-reads flagged articles, decides ship/block manually

LLM rating is a cheap secondary signal, not a hard gate. Manual scoring is the authority.

### Testing strategy

| File | Coverage |
|------|----------|
| `test_topic_cluster.py` | Lazy singleton — import does not trigger model load. Deterministic seed (`cluster_posts(..., seed=42)`); 20 synthetic posts → expected cluster count + centroid shape + label format. Empty/small input fallbacks. |
| `test_source_baseline.py` | Sumy LexRank deterministic on known text. Abstractive mocked, structure asserted. All-3-fail fallback. |
| `test_correlator.py` | Mock Gemini for 3 baselines, assert DeltaRecord shape + Literal status values. |
| `test_persona_v2.py` | v1 file → loader raises `unsupported_persona_schema`. v2 file → topics populated. Round-trip serialize. Full clustered post text preserved across save/load. Incomplete-persona round-trip. |
| `test_generation_blend.py` | Synthetic centroids + mock embed singleton → assert softmax weights + blended prompt content. Embedding-model-mismatch raises with exit code 3. `_global`-only persona skips per-topic blend path. PromptBuilder imports nothing from `sentence_transformers` / `bertopic` (purity-invariant assertion). |
| `test_synthesize_style_v2.py` | End-to-end with mocked Gemini + small post fixture → persona file emitted with topics populated. Verify JSON error envelope on simulated baseline failure. Partial-write on simulated exit-4 leaves `status: incomplete`. |
| `test_scrape_filter.py` | Regex matches "ignore previous instructions", `<\|im_start\|>`, "you are now". NFKC normalization catches homoglyphs (e.g., Cyrillic "і" → Latin "i"). Zero-width stripping catches `i​gnore`. Non-matches pass through. Pattern fixture file is loaded, not hardcoded. |
| `test_error_contract.py` | Each failure mode → expected JSON envelope shape + exit code. Stderr-vs-stdout channel separation asserted. Envelope serialization strips absolute paths, `os.environ` values, and matched injection content (regression guards). |
| `test_corpus_sampler.py` | Last-year filter, topic-balanced sampling honors floor=10/topic where corpus permits, per-topic eval holdout (N_judge + N_manual) is disjoint from training set. CLI flag defaults wire through. |

LLM calls always mocked in tests. Real Gemini hit only via the live-validation smoke test (separate, opt-in).

### Sentence-transformers singleton

`tldr_scholar/topic_cluster.py` — **lazy load on first call**. `tldr-scholar --help`, `list-personas`, config inspection, and any path that never embeds incur zero model-load cost.

```python
_MODEL: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL

def embed_text(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()

def embed_batch(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, normalize_embeddings=True).tolist()
```

Module is safe to `import topic_cluster` without triggering download. Tests mock `_get_model` via monkeypatch. PromptBuilder imports `embed_text` directly; remains a pure-string builder otherwise. Import direction is one-way: `prompts.py → topic_cluster`. `topic_cluster` MUST NOT import from `prompts` (cycle prevention).

### Migration / rollout

**Single epic, full pipeline scope.** All work units ship together (per design decision: no MVP cut). Internal sequencing of work units within the epic:

1. **WU-1: cleanup + schema.** Delete v1 personas + v1 schema fields. Add `topic_cluster.py` with **lazy** singleton + bertopic labeling. Refactor `personas.py` to v2-only with mandatory `topics` + `embedding_model` + optional `status`/`incomplete_stages`. Add `scrape_filter.py` with NFKC normalization + zero-width strip + regex over fixture `tests/fixtures/injection_patterns.txt`.
2. **WU-2: corpus sampler.** Implement last-year window scrape + topic-balanced N_train sampling. Hold out per-topic eval sets (N_judge, N_manual). CLI flags `--window-months`, `--n-train`, `--n-judge-per-topic`, `--n-manual-per-topic`.
3. **WU-3: baselines + correlation.** Add `source_baseline.py`, `correlator.py`. Wire 3-baseline pipeline behind `--full-baselines` flag (default: claims-only). Add `NEUTRAL_SUMMARY_PROMPT` with `<untrusted_content>` delimiter.
4. **WU-4: aggregation.** Per-topic LLM aggregation (1 call/topic, full aggregation contract). Global synthesis runs over all DeltaRecord. Populate TopicProfile + Persona top-level.
5. **WU-5a: error contract.** JSON envelope emitter + exit codes table. Content-echo policy enforced (no matched content, no abs paths, no env). End-of-run drop summary on stdout. Partial-write semantics for exit code 4 (persona `status: incomplete` + `incomplete_stages`).
6. **WU-5b: CLI flag wiring.** Add `--full-baselines`, `--reset=<stage>`, `--window-months`, `--n-train`, `--n-judge-per-topic`, `--n-manual-per-topic`. CorpusCache integration for stage cache keys.
7. **WU-6: generation blending.** Modify `prompts.py` to detect topics + blend via softmax. When `topics` empty or `_global`-only: skip softmax blend, use top-level revelation_priorities/suppression_rules directly. Remove flat-priority path entirely. Wire embedding-model-mismatch → exit code 3.
8. **WU-7: tests.** All 9 test files per testing strategy table.
9. **WU-8: live validation + ship-gate.** Run pipeline against `https://fediscience.org/@davdittrich`. Execute evaluation protocol (manual + LLM-as-judge). Apply ship-gate triage rule. Block ship if regression suspected.

### Live validation

Real-world dataset: `https://fediscience.org/@davdittrich`.

1. After WU-7 lands: `tldr-scholar-synthesize-style https://fediscience.org/@davdittrich --name davd --full-baselines` → v2 persona with topics + centroids populated.
2. Run 10 sample articles (mix of econ + tech + behavioral) through v1 (archived baseline) and v2.
3. Score per evaluation protocol (manual + LLM-as-judge).
4. Inspect persona YAML: topic labels meaningful (top-3 c-TF-IDF reflect real themes), centroid count matches HDBSCAN output, full post text preserved.
5. Latency budget per gen call < 2s (embed + similarity is cheap).

### Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| 3.8x LLM cost when --full-baselines invoked | High | Opt-in; CorpusCache reuses baselines across re-runs |
| HDBSCAN tuning across feed types | Medium | `_global` fallback; `--min-cluster` flag |
| sentence-transformers model download blocks first run | Medium | Pre-download via post-install hook; pre-flight check warns if missing |
| Centroid drift if model is updated externally | Medium | `embedding_model` field; mismatch → exit 3 |
| Auto-label quality from c-TF-IDF | Low | Labels are advisory; centroids drive matching |
| Generation latency from embedding + similarity | Low | One embed call (~50ms), N cosine sims (<1ms); negligible |
| Prompt injection from scraped post content | Medium | Regex pre-filter at scrape + `<untrusted_content>` delimiter at prompt + LLM data-not-instructions instruction |
| Full post text on disk persists hostile content across runs | Medium | Same regex filter applied before persistence; injected posts never reach YAML |
| Eval protocol false-positive (LLM judge bias) | Low | Manual scoring is gating; LLM judge is corroborating |

## Resolved decisions (design-review-gate rounds 1 + 2)

| Item | Decision |
|------|----------|
| Motivation | User feedback: summaries off-topic / under-personalized |
| Corpus sampling | Last-year window; cluster → topic-balanced N_train=200; per-topic eval holdouts (N_judge=10, N_manual=5); all configurable |
| Evaluation gate | Manual mean ≥4.0 (gating) + LLM intrinsic 1-5 rating ≥4.0 (advisory triage). NO v1 comparison (v1 deleted, no archive) |
| Cost default | `--full-baselines` opt-in; claims-only default |
| c-TF-IDF impl | `bertopic` dep |
| v1 personas | Discontinued. Delete files. Loader v2-only, no migration |
| Per-topic aggregation | 1 Gemini call per topic, full aggregation (no local-freq alt) |
| YAML content | Full clustered post text persisted in TopicProfile.posts |
| Injection defense | Scrape-time NFKC + zero-width-strip + regex (fixture file) + prompt-time `<untrusted_content>` delimiter |
| Embedding model load | Module-level singleton, **lazy** on first call, free function (not on PromptBuilder); one-way import direction prompts → topic_cluster |
| Error contract | JSON envelope on stderr; no content/path/env echo; distinct exit codes; end-of-run drop summary on stdout |
| Partial writes (exit 4) | Persist with `status: incomplete` + `incomplete_stages`; resume via `--reset=<stage>` |
| Re-run state | `--reset=<cluster|correlate|aggregate|all>` granular flag, default preserve all |
| DeltaRecord.status | `Literal[str]` (not Pydantic Enum) |
| WU-5 scope | Split: WU-5a error contract + WU-5b CLI flags |
| Epic scope | Full pipeline + abstractive baseline, single epic, 9 WUs |

## Out of scope

- Persona generation from posts without linked sources (skip-links mode)
- Per-language persona handling
- Persona versioning / migration tooling (v2 is terminal; future schema breaks delete + regen)
- Multi-persona blending at generation time
