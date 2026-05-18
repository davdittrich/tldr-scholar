# Persona Neutral-Baseline Delta — Design

**Date:** 2026-05-18
**Status:** Draft — pending design-review-gate
**Author:** persona-creator brainstorm session
**Related epic:** tldr-scholar-0pb (Improve Generated Text Quality)

## Problem

Current `tldr-scholar-synthesize-style` pipeline extracts a persona by comparing each social post against atomic claims decomposed from the linked source. This catches per-claim shared/suppressed/distorted patterns, but:

1. **Loses framing/emphasis signal.** Atomic claims discard the abstraction level a neutral summary preserves (e.g., a paper claim "X causes Y, N=50" decomposes to flat claims; the SUMMARY would have bundled the N=50 as a qualifier — persona dropping the qualifier shows up as framing, not claim omission).
2. **Per-post deltas only, no per-topic aggregation.** Patterns aggregate globally into one flat agenda. A persona that suppresses statistics in econ posts but shares them in tech posts gets averaged into noise.
3. **LLM-driven topic classification.** Gemini `CLASSIFICATION_PROMPT` is expensive, non-deterministic, and provides no embeddings for downstream similarity at generation time.
4. **Generation doesn't mirror the selective-emphasis pattern.** When `tldr-scholar` uses a persona to summarize a new article, it applies a global priority list. It cannot match the article's topic to the persona's per-topic emphasis pattern.

## Goal

Replace classification + correlation stages with a local-NLP + multi-baseline pipeline that captures both per-claim AND framing-level deltas, aggregates per-topic, and emits per-topic centroids so generation can blend topic profiles by article-to-centroid similarity.

## Non-goals

- Multi-author persona blending
- Persona persistence beyond YAML files
- Real-time persona refinement during generation
- Replacing Gemini with a local LLM for summarization

## Design overview (Approach B — pipeline replacement)

```
posts (list[SocialPost])
    │
    ▼
[topic_cluster.py]
embed (sentence-transformers all-MiniLM-L6-v2) → HDBSCAN cluster → labels + centroids
    │
    ▼
posts_by_topic: dict[topic_label, list[SocialPost]]
    │
    ▼ (per post + matched source)
[source_baseline.py]
3 baselines per source:
  1. atomic claims (Gemini DECOMPOSITION_PROMPT — existing)
  2. extractive 5-sentence summary (sumy LexRank — local, deterministic)
  3. abstractive 3-sentence neutral summary (Gemini NEUTRAL_SUMMARY_PROMPT — new)
    │
    ▼
[correlator.py]
3 Gemini correlation calls per (post, source), one per baseline →
delta_record = {baseline_type, statements, status_per_statement, intent}
    │
    ▼ (aggregate per topic)
TopicProfile per cluster
    │
    ▼ (aggregate global)
Persona (top-level fields) + topics: dict[label, TopicProfile] + topic_centroids
```

### Module structure

**New:**
- `tldr_scholar/topic_cluster.py` — `cluster_posts(posts) -> (labels: list[str], centroids: dict[str, list[float]])`
- `tldr_scholar/source_baseline.py` — `class SourceBaselines: claims, extractive_summary, abstractive_summary`
- `tldr_scholar/correlator.py` — `correlate_against_baselines(post, baselines) -> list[DeltaRecord]`

**Modified:**
- `tldr_scholar/synthesize_style.py` — orchestrates the new pipeline. Drops `CLASSIFICATION_PROMPT`. Adds `NEUTRAL_SUMMARY_PROMPT`. Replaces `correlate_post_to_source` invocation with `correlator.correlate_against_baselines`.
- `tldr_scholar/personas.py` — adds `TopicProfile` Pydantic model. `Persona` gains `topics: dict[str, TopicProfile]` and `embedding_model: str | None`.
- `tldr_scholar/prompts.py` — `PromptBuilder.build_system_prompt` for persona path detects non-empty `topics`, embeds the input source, computes softmax-weighted blend over top-K topics, builds blended `revelation_priorities` / `suppression_rules` text.

**New deps (pyproject.toml runtime):**
- `sentence-transformers>=2.2`
- `hdbscan>=0.8`

`sumy` already present.

### Persona schema v2

```python
class TopicProfile(BaseModel):
    label: str                              # human-readable, auto-labeled by top-3 c-TF-IDF terms
    centroid: list[float]                   # 384-d for all-MiniLM-L6-v2 (normalized)
    sample_size: int                        # # posts in cluster
    revelation_priorities: list[str] = Field(default_factory=list)
    suppression_rules: list[str] = Field(default_factory=list)
    substantive_anchors: list[str] = Field(default_factory=list)
    rhetorical_strategy: str | None = None
    confidence: int = 0                     # 0-100, aggregated from baseline agreement

class Persona(BaseModel):
    # Core identity (unchanged)
    name: str
    role: str
    tone: str
    structure_pattern: str
    hashtag_style: str = "lowercase"
    # Global (aggregated across topics)
    agenda: str | None = None
    worldview: str | None = None
    pivot_logic: str | None = None
    identifiable_nuances: list[str] = Field(default_factory=list)
    attribute_confidence: dict[str, int] = Field(default_factory=dict)
    # Per-topic (mandatory in v2; min 1 topic — fallback "_global" if clustering yields nothing)
    topics: dict[str, TopicProfile]
    embedding_model: str
```

**No backward compat with v1 personas.** Existing files (`davd.yaml`, `davdittrich.yaml`, `samples.yaml`) are dropped (user authorized: "old personas can be deleted. they are shit"). Old flat fields `revelation_priorities`/`suppression_rules`/`substantive_anchors`/`rhetorical_strategy` removed at top level — these now live exclusively inside TopicProfile. Global keeps only `agenda`/`worldview`/`pivot_logic`/`identifiable_nuances` which are cross-cutting by nature.

### Topic clustering details

- Model: `sentence-transformers/all-MiniLM-L6-v2` (~50MB, CPU, 384-d output)
- Embed batch: all sampled posts at once
- HDBSCAN(min_cluster_size=5, min_samples=3, metric='cosine')
- Noise points (label = -1) → bucketed under `"_unclustered"` topic
- If HDBSCAN yields 0 real clusters → fallback single `"_global"` topic containing all posts
- Centroid = mean of normalized embeddings within cluster, re-normalized
- Auto-labeling: c-TF-IDF (class-based TF-IDF) → top-3 terms per cluster joined `+`. Example: `"economics+labor+wage"`. No LLM call required; deterministic.

### Baseline generation

Per matched (post, source) pair:

| Baseline | Source | Compute | Deterministic |
| -------- | ------ | ------- | ------------- |
| atomic claims | Gemini `DECOMPOSITION_PROMPT` (existing) | 1 LLM call | no |
| extractive summary | sumy `LexRankSummarizer`, 5 sentences | local CPU | yes |
| abstractive summary | Gemini `NEUTRAL_SUMMARY_PROMPT` (new), 3 sentences | 1 LLM call | no |

**`NEUTRAL_SUMMARY_PROMPT`:**
```
Produce a neutral 3-sentence summary of the following text. Report findings only.
No opinion, no agenda, no editorial framing. Preserve numerical specifics
(N, p-values, percentages) when present.

Text:
{text}

Return ONLY the 3-sentence summary as plain text. No headers, no bullets.
```

### Correlation

Per (post, baselines) triple:

```python
class DeltaRecord(BaseModel):
    baseline_type: Literal["claims", "extractive", "abstractive"]
    statements: list[str]                  # the baseline as a list of comparable units
    status_per_statement: list[Literal["shared", "suppressed", "distorted"]]
    intent: str | None                     # one-sentence Gemini-extracted intent
```

For claims baseline: re-use existing CORRELATION_PROMPT (with claims as statements input).
For extractive: each sentence is one statement.
For abstractive: each sentence is one statement.

Output: 3 DeltaRecord per (post, source).

### Per-topic aggregation

For each topic, collect all DeltaRecord across all posts in cluster. Aggregate:

- `revelation_priorities`: top-N statements with `status="shared"` across all baseline types, weighted by frequency.
- `suppression_rules`: top-N with `status="suppressed"`.
- `substantive_anchors`: top-N statements with `status="distorted"` (signals re-framing).
- `rhetorical_strategy`: LLM call with all intent strings → 1 sentence.
- `confidence`: percentage of baselines agreeing per attribute.

Single Gemini call per topic for the aggregation (or local frequency analysis for the lists, LLM only for rhetorical_strategy).

### Global aggregation

Run existing `DEEP_SYNTHESIS_PROMPT` over all DeltaRecord (across all topics) → populates Persona top-level `agenda` / `worldview` / `pivot_logic` / `identifiable_nuances`.

### Generation blending

In `prompts.py` `PromptBuilder.build_system_prompt` persona path:

1. Load `sentence-transformers` model named in `p_config.embedding_model`.
2. Embed input source text → 384-d normalized vector.
3. For each topic: `cosine_sim = dot(input_emb, centroid)`.
4. Softmax with temperature τ=0.3 over similarities → weight per topic.
5. Threshold: drop topics with weight < 0.1.
6. Top-K=3 retained topics.
7. Build blended priority/suppression lists: weighted-frequency vote across retained TopicProfiles.
8. Inject into existing PERSONA_SYSTEM_PROMPT placeholders (no template change).

**Embedding model mismatch guard:** if `p_config.embedding_model` ≠ locally installed default → log error and raise. No fallback — v1 schema is gone, every v2 persona must declare its model. Mismatch indicates corrupted persona or model rollback; should not silently degrade.

### Error handling

| Failure | Behavior |
| ------- | -------- |
| sentence-transformers model download fails | log error, fall back to single `"_global"` topic (no clustering) |
| HDBSCAN yields 0 clusters | single `"_global"` topic |
| sumy summary fails | drop extractive baseline for that source, continue with 2 baselines |
| Gemini abstractive fails | drop abstractive baseline, continue with claims + extractive |
| Gemini correlation fails | log warning, drop that DeltaRecord |
| All baselines fail for a source | drop the source from corpus |
| Centroid model mismatch at generation | log error, raise — v2 always declares model |

### Compute budget

Per persona run with ~200 sampled posts, ~150 with linked sources:

| Stage | LLM calls | Local CPU |
| ----- | --------- | --------- |
| Embed posts | 0 | ~0.5s |
| Cluster | 0 | <1s |
| Decompose (claims) | 150 | — |
| Abstractive summary | 150 | — |
| Extractive summary | 0 | <2s |
| Correlate (3 baselines × 150) | 450 | — |
| Per-topic aggregate | ~5-10 (one per topic) | — |
| Global synthesis | 1 | — |
| **Total** | **~755-760 LLM calls** | **~5s** |

Current pipeline: ~200 LLM calls. **Increase: ~3.8x.** Mitigation: `--no-baselines` flag for cheap re-runs that skip abstractive + correlation (uses only claims baseline like current pipeline).

### Backward compatibility

**None.** Old persona files (`davd.yaml`, `davdittrich.yaml`, `samples.yaml`) are deleted in phase 1 of the rollout — user authorized cleanup ("old personas can be deleted. they are shit"). v2 schema removes obsolete top-level fields (`revelation_priorities`, `suppression_rules`, `substantive_anchors`, `rhetorical_strategy`) which now exclusively live in TopicProfile.

Generation no longer needs flat-priority fallback path. PersonaManager rejects v1 files at load time with a clear error pointing to the regenerate-via-synthesize-style flow.

### Embedding model mismatch

| Failure | Behavior |
| ------- | -------- |
| Centroid model mismatch at generation | log error, raise. v2 always declares model; mismatch is corruption, fail loud. |

### Testing strategy

| Test file | Coverage |
| --------- | -------- |
| `test_topic_cluster.py` | Deterministic seed; 20 synthetic posts → expected cluster count + centroid shape + label format. Empty/small input fallbacks. |
| `test_source_baseline.py` | Sumy LexRank deterministic on known text. Abstractive mocked, structure asserted. All-3-fail fallback. |
| `test_correlator.py` | Mock Gemini for 3 baselines, assert DeltaRecord shape + status enum. |
| `test_persona_v2.py` | Load existing flat persona → empty topics. Load new topic persona → topics populated. Round-trip serialize. |
| `test_generation_blend.py` | Synthetic centroids + mock sentence-transformers → assert softmax weights + blended prompt content. Mismatch model fallback. |
| `test_synthesize_style_v2.py` | End-to-end with mocked Gemini + small post fixture → persona file emitted with topics populated. |

### Migration / rollout

3-phase epic structure for safer rollout:

**Phase 1: drop v1 + clustering + centroids skeleton**
- Delete `davd.yaml`, `davdittrich.yaml`, `samples.yaml` from `~/.config/tldr-scholar/personas/` (user-managed dir; commit a `.gitkeep` only)
- Update tests/fixtures that reference v1 personas to use generated v2 fixtures or skip with reference to this design
- Add `topic_cluster.py`
- Refactor `personas.py`: drop v1 fields, add `TopicProfile`, make `topics` and `embedding_model` mandatory
- Modify `synthesize_style.py`: cluster posts, store centroids, write topics field. No per-topic correlation yet — TopicProfile populated with cluster metadata only (empty priority lists).
- Persona file gains topic skeleton (centroids only). Validates clustering + new schema on real data without correlation cost.

**Phase 2: multi-baseline correlation + per-topic aggregation**
- Add `source_baseline.py`, `correlator.py`
- Add `NEUTRAL_SUMMARY_PROMPT`
- Wire into `synthesize_style`
- Populate TopicProfile.revelation_priorities/suppression_rules/substantive_anchors per topic
- Adds the ~3.8x LLM call cost

**Phase 3: generation blending**
- Modify `prompts.py` to detect topics + blend
- Remove flat-priority generation path entirely
- Add embedding-model-mismatch raise

Each phase ships as its own epic; phase 2 depends on phase 1, phase 3 depends on phase 2.

### Acceptance test (end-to-end)

Real-world dataset: `https://fediscience.org/@davdittrich` (used in v4 development).

1. After phase 3 lands: run `tldr-scholar-synthesize-style https://fediscience.org/@davdittrich --name davd` → v2 persona with topics + centroids populated.
2. For 5 fixed sample articles (mix of econ + tech + behavioral), run `tldr-scholar --persona davd` against each. Manually inspect output for:
   - Per-topic emphasis differences (econ output emphasizes different aspects than tech output for the persona)
   - Coherent voice (no regression to incoherent output)
   - Latency increase per generation call < 2s (embed + similarity is cheap)
3. Inspect generated persona YAML: verify topic labels are meaningful (top-3 c-TF-IDF terms reflect real themes), centroid count matches HDBSCAN output.

## Risks

| Risk | Severity | Mitigation |
| ---- | -------- | ---------- |
| 3.8x LLM cost balloon | High | `--no-baselines` flag; baselines computed only once per source via CorpusCache extension |
| HDBSCAN tuning across feed types | Medium | Falls back to `_global` topic gracefully; min_cluster_size configurable via CLI flag `--min-cluster` |
| sentence-transformers model download blocks first run | Medium | Pre-download in qe4.11-style declare step; pre-flight check warns if missing |
| Centroid drift if model is updated externally | Medium | `embedding_model` field on Persona; mismatch → log + flat fallback |
| Auto-label quality from c-TF-IDF | Low | Labels are advisory; centroids drive matching, not labels |
| Generation latency from embedding + similarity | Low | One embed call (~50ms), N cosine sims (<1ms); negligible |

## Open questions (resolved during brainstorm)

- Baseline scope: **both atomic claims AND neutral summary (extractive + abstractive)**
- Topic ID method: **sentence-transformers + HDBSCAN**
- Persona shape: **topic-level breakdown plus global**
- Scope: **extraction + generation with weighted blend**

## Out of scope

- Persona generation from posts without linked sources (skip-links mode)
- Per-language persona handling
- Persona versioning / migration tooling
- Multi-persona blending at generation time
