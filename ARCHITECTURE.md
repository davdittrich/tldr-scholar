# Architecture

`tldr-scholar` is a modular summarization pipeline designed for academic and general-purpose text. It prioritizes low-latency, privacy (via local LLMs), and audience-specific personalization.

## Core Components

### 1. Ingestion (`ingest.py`)
Responsible for converting various inputs (PDF, URL, Markdown) into clean, plain text.
- Uses `PyMuPDF` for local PDFs.
- Uses `curl-cffi` + `trafilatura` for URLs, with a fallback chain to Open Access providers (Unpaywall, OpenAlex).

### 2. Persona Management (`personas.py`)
Loads writing style profiles from YAML files in `~/.config/tldr-scholar/personas/`.
- Allows for dynamic style synthesis without modifying the core codebase.
- Personas define `role`, `tone`, `structure_pattern`, and `hashtag_style`.

### 3. Prompt Engineering (`prompts.py`)
The `PromptBuilder` class centralizes all instruction-building logic.
- Implements the **IMRAD-aware scientific structure** for academic papers.
- Applies audience-specific templates (Expert vs. Layman).
- Handles persona-specific structural overrides (e.g., "Stitched Quotes").
- Includes quality guardrails for short text and missing metadata.

### 4. Backend Dispatch (`backends/`)
A tiered execution system with an optional fallback chain.
- **Gemini**: Cloud-based summarization via the ACP protocol.
- **Lemonade**: Local summarization via an OpenAI-compatible API.
- **Ollama**: Local summarization via the Ollama API.
- **Extractive**: Rule-based summarization using LexRank (no LLM required).

### 5. Hashtag Generation (`hashtags.py`)
Generates social-media-ready tags.
- LLM backends parse hashtags directly from the response.
- Extractive backend falls back to a TF-IDF heuristic with **bigram support** to capture multi-word technical terms.
- Supports both `lowercase` and `PascalCase` formatting.

## Data Flow

1.  **CLI/API** receives a source and parameters (audience, tone, length).
2.  **Ingestion** extracts text and identifies the input type.
3.  **PersonaManager** loads the requested style if applicable.
4.  **PromptBuilder** assembles the system and user prompts.
5.  **Backend** executes the summarization (with fallback if enabled).
6.  **Hashtag Generator** extracts or derives tags.
7.  **SummaryResult** is returned and formatted for output (Text, JSON, or Markdown).
