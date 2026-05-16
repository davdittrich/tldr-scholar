# Design: Dynamic Agendas & Deep Personas

## 1. Problem Statement
Current personas focus on *surface style* (tone, role, formatting). They lack *depth of intent*—the agenda, worldview, and specific extraction filters that define a real human expert's writing.

## 2. Implementation in tldr-scholar (Generating Stage)

### Schema Expansion (`models.py` & `personas.py`)
Add fields to `Persona` model:
- `agenda`: High-level purpose of the persona's writing (e.g., "Demystify corporate capture of academia").
- `worldview`: Implied philosophical/political leaning.
- `extraction_filter`: Explicit instructions on what to prioritize (e.g., "Always extract funding source and p-values") and what to ignore (e.g., "Ignore 'future work' fluff").
- `persuasion_goal`: What the persona wants to convince the reader of.

### Prompt Logic Updates (`prompts.py`)
Refactor `PromptBuilder` to inject these deep layers:
1. **The Lens**: Instructions on how to *read* the text through the persona's worldview.
2. **The Sieve**: Explicit extraction/exclusion rules based on `extraction_filter`.
3. **The Hook**: Structure the summary to lead toward the `persuasion_goal`.

## 3. Generating Deep Personas (Synthesis Stage)

### Multi-Source Analysis (`bin/synthesize_style.py`)
Upgrade the synthesis script to use a "Deep Persona Chain":
1. **Analysis Pass 1 (Style)**: Tone, vocabulary, linguistic nuances.
2. **Analysis Pass 2 (Intent)**: Identify recurring arguments, "enemies," and preferred evidence types.
3. **Synthesis**: Generate the expanded YAML.

### Interactive Refinement (Clarity Interviews)
A new command `tldr-scholar refine-persona` that:
1. Presents the synthesized profile.
2. Asks targeted "Interview Questions" to fill gaps in worldview or extraction priorities.
3. Updates the YAML based on user feedback.

## 4. Work Units
1. **WU-1**: Expand `Persona` Pydantic model with deep intent fields.
2. **WU-2**: Update `PromptBuilder` to use intent fields for "Lensed Extraction."
3. **WU-3**: Upgrade `synthesize_style.py` with intent-detection prompts.
4. **WU-4**: Implement `refine-persona` CLI logic.
