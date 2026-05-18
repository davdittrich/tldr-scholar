# Active Plan
<!-- approved: 2026-05-18 -->
<!-- gate-iterations: 2 -->
<!-- user-approved: true -->
<!-- status: in-progress -->
<!-- epic: tldr-scholar-nwl -->
<!-- parent-feature: tldr-scholar-0pb -->
<!-- supersedes: epic-plan-v3 (qe4/zaf/4lg all closed) -->

# Epic Plan — Persona Neutral-Baseline Delta (topic-blended generation)

## Epic
`tldr-scholar-nwl` (P2, 9 child WUs)

## Design Spec
`docs/superpowers/specs/2026-05-18-persona-neutral-baseline-delta-design.md`
- design-review-gate: APPROVED by PM/Architect/Designer/Security/CTO (round 2)
- plan-review-gate: PASS Feasibility/Completeness/Scope (round 2)

## Work Unit DAG
- WU-1 (nwl.1) — cleanup + v2 schema + lazy embedding singleton + scrape_filter + preflight + relocate prompts + delete refine_persona + migrate v1 tests. **READY.**
- WU-2 (nwl.2) — corpus sampler + wire injection filter + exit-5 emit. Deps: WU-1.
- WU-3 (nwl.3) — baselines (claims/extractive/abstractive) + correlation + untrusted_content delimiter. Deps: WU-1, WU-2.
- WU-4 (nwl.4) — per-topic + global aggregation (1 LLM call/topic, posts persisted). Deps: WU-3.
- WU-5a (nwl.5) — JSON error envelope + exit codes + partial-write incomplete flag. Deps: WU-1.
- WU-5b (nwl.6) — CLI flags (--full-baselines, --reset=<stage>, sampler flags, --min-cluster) + CorpusCache per-stage redesign. Deps: WU-5a, WU-2.
- WU-6 (nwl.7) — generation blending (softmax, _global skip, mismatch exit 3). Deps: WU-1, WU-4.
- WU-7 (nwl.8) — full test suite (11 files) + flat-priority dead-code assertion + preflight invocation assertion. Deps: WU-1..WU-6.
- WU-8 (nwl.9) — live validation + ship-gate (manual gating + LLM intrinsic advisory, --max-llm-calls fence). Deps: WU-7.

## Ship-Gate Rule (revised — no v1 comparison)
- Manual mean ≥4.0 on topical relevance + style fidelity (gating)
- LLM intrinsic 1-5 rating mean ≥4.0 (advisory)
- Manual fails → BLOCK regardless
- Manual passes + LLM <4.0 → author-resolved triage

## Key Decisions (rounds 1 + 2 design-gate, plan-gate round 1 fixes)
- v1 schema discontinued, no migration, refine_persona.py deleted
- `--full-baselines` opt-in (default = claims-only, current cost)
- `bertopic` runtime dep for c-TF-IDF topic labels
- Python pin deferred — WU-1 documents wheel-availability outcome
- Sentence-transformers singleton in `topic_cluster.py`, lazy load, free fn
- DeltaRecord.status = Literal[str] (Pydantic Enum gotcha)
- Full clustered post text persisted in TopicProfile.posts
- Injection defense: NFKC normalize + zero-width strip + regex over fixture + `<untrusted_content>` delimiter in all external-content prompts
- JSON error envelope on stderr, content-echo policy (no abs paths/env/matched-content/signed-URL query strings)
- Exit codes: 0 ok, 1 internal, 2 v2 schema fail, 3 embedding mismatch, 4 LLM exhausted (partial-write with status:incomplete), 5 empty corpus
- `--reset=<cluster|correlate|aggregate|all>` granular flag, default preserve

## Acceptance
- [ ] All 9 child WUs closed
- [ ] `tldr-scholar-synthesize-style https://fediscience.org/@davdittrich --full-baselines` runs end-to-end
- [ ] Validation report committed under `docs/superpowers/validation/`
- [ ] No v1 persona files remain on disk
- [ ] No flat-priority code path remains (AST assertion passes)
- [ ] PromptBuilder purity preserved (no sentence_transformers/bertopic imports)
- [ ] All commits reference epic + WU IDs
- [ ] `bd preflight` passes
