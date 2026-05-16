# Implementation Plan: Hardened Analytic Pipeline & Architectural Cleanup (v2)

## Goal
Standardize the codebase, ensure data integrity for personas, and fully implement the bottom-up atomic analytic engine.

## Mechanism
- **Types Separation**: Break circularity by moving shared Enums to `tldr_scholar/types.py`.
- **Validation Gate**: Mandatory Pydantic validation (`Persona.model_validate`) for all persona updates.
- **Atomic Pipeline**: Decompose -> Correlate -> Synthesize loop.
- **Logging**: Transition to `loguru` for structured output.

## Phase 1: Architectural Cleanup (Tickets 655.4, 655.5, 655.3)
- Create `tldr_scholar/types.py`; move `AudienceEnum`, `ToneEnum`.
- Centralize `DEFAULT_PERSONA_DIR` in `config.py`.
- Update `prompts.py` to move imports to top level.

## Phase 2: Data Integrity & Validation (Tickets 655.2, pk6.4, pk6.5)
- Implement `deep_merge` in `refine_persona.py`.
- Add Pydantic validation before YAML save in `refine_persona.py`.
- Replace `print()` with `logger` in `synthesize_style.py` and `refine_persona.py`.

## Phase 3: Atomic Engine Integration (Tickets 655.1, 8o6.3, 8o6.4)
- **TDD Hook**: Add unit tests in `tests/test_synthesis.py` specifically for `decompose_source` and `correlate_post_to_source` logic.
- Wire `decompose` -> `correlate` -> `synthesize` loop in `synthesize_style.py`.
- **Gap Detection**: Implement `detect_profile_gaps` using `DOMAIN_GAP_DETECTION_PROMPT`, comparing the persona's `stance_space` (agenda, worldview, etc.) against a role-based domain map.

## Forbidden
- No method-level imports for core modules.
- No unvalidated YAML writes.

## Audit Strategy
- `pytest`: All 64+ existing tests + new atomic logic tests must pass.
- `tldr-scholar synthesize-style`: Verify output YAML has `agenda` and `worldview`.
- `tldr-scholar refine-persona`: Verify LLM-driven gap detection identifies missing substantive anchors.
