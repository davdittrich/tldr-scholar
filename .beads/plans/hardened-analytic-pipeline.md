# Implementation Plan: Hardened Analytic Pipeline & Architectural Cleanup

## Goal
Standardize the codebase, ensure data integrity for personas, and fully implement the bottom-up atomic analytic engine.

## Mechanism
- **Types Separation**: Break circularity by moving shared Enums to `tldr_scholar/types.py`.
- **Validation Gate**: Mandatory Pydantic validation (`Persona.model_validate`) for all persona updates.
- **Atomic Pipeline**: Decompose -> Correlate -> Synthesize loop as the primary engine for style analysis.
- **Logging**: Full transition to `loguru` for structured output.

## Phase 1: Architectural Cleanup (Tickets 655.4, 655.5, 655.3)
- Create `tldr_scholar/types.py` and move `AudienceEnum`, `ToneEnum`.
- Centralize `DEFAULT_PERSONA_DIR` in `config.py`.
- Update `prompts.py` to move imports to top level.
- Tighten type hints in `__init__.py`.

## Phase 2: Data Integrity & Validation (Tickets 655.2, pk6.4, pk6.5)
- Implement `deep_merge` in `refine_persona.py`.
- Add Pydantic validation before YAML save in `refine_persona.py`.
- Replace `print()` with `logger` in `synthesize_style.py` and `refine_persona.py`.

## Phase 3: Atomic Engine Integration (Tickets 655.1, 8o6.3, 8o6.4)
- Wire `decompose` -> `correlate` -> `synthesize` loop in `synthesize_style.py`.
- Make atomic analysis the default for all corpus formats.
- Implement real LLM-based gap detection in `refine_persona.py` using domain baselines.

## Forbidden
- No method-level imports for core modules.
- No unvalidated YAML writes.
- No `print()` statements in production code.

## Audit Strategy
- `pytest`: All 64+ existing tests must pass.
- `tldr-scholar synthesize-style`: Verify persona generation includes `agenda` and `worldview`.
- `tldr-scholar refine-persona`: Verify interactive gap detection returns valid substantive topics.
