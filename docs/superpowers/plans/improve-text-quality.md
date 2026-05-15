# Design Document: Improving Generated Text Quality

## Problem Statement
The current summarization output is functional but lacks audience-specific personalization and highly relevant hashtags. The scientific structure is rigid, and non-academic content could benefit from better intent detection.

## Proposed Changes

### 1. Audience-Specific Personalization
Add an `audience` parameter to `SummaryRequest` to allow users to specify who the summary is for (e.g., "expert", "layman", "student").

- **Models**: Update `SummaryRequest` in `models.py`.
- **Prompts**: Update `SCIENTIFIC_SYSTEM_PROMPT` in `prompts.py` to use `{audience_instruction}`.
- **Logic**: Add a helper in `prompts.py` to generate instructions like "Explain concepts as if teaching someone with a PhD in this field" vs. "Use simple analogies for a high school student".
- **CLI**: Add `--audience` (expert|layman|student) to `cli.py`.

### 2. Improved Scientific Structure
Refine the `_SCIENTIFIC_SENTENCES` to be more descriptive and ensure the "Layman" mode actually uses more analogies.

### 3. Dynamic Hashtag Generation
Improve `hashtags.py` by:
- Providing better examples in the instruction.
- Enhancing the TF-IDF fallback to prioritize multi-word technical terms (e.g., "machine learning" instead of "machine", "learning").

### 4. General Mode Enhancements
Update `GENERAL_SYSTEM_PROMPT` to include a "tone" instruction (e.g., professional, casual, analytical).

## Impact
- **Personalization**: Summaries will better match the reader's expertise level.
- **Informative**: Clearer focus on impact vs. results based on audience.
- **Appeal**: Better hashtags and formatting for social sharing.

## Work Units (DRAFT)
1. Update `models.py` and `SummaryRequest`.
2. Implement audience/tone logic in `prompts.py`.
3. Update `cli.py` with new options and wiring.
4. Refine `hashtags.py` logic.
5. Update tests and README.
