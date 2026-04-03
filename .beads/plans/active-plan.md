---
title: "tldr-scholar: Clean code pass (8 issues)"
status: in-progress
created: 2026-04-03
work_units: 2
baseline_tests: 64
---

# tldr-scholar Clean Code Pass

## Issues

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | REQUIRED | backends/lemonade.py | System-only message violates OpenAI chat spec |
| 2 | REQUIRED | ingest.py | Unused `Optional` import |
| 3 | REQUIRED | cli.py | Unused `SummaryRequest` import |
| 4 | REQUIRED | cli.py | Lazy `urlparse` import inside function |
| 5 | SUGGESTION | ingest.py | PDF extraction duplicated between `_ingest_pdf` and `_ingest_url` |
| 6 | SUGGESTION | config.py | Unused `Any`, `Optional` imports |
| 7 | SUGGESTION | hashtags.py | Unused `Optional` import |
| 8 | SUGGESTION | backends/ollama.py | Unused `logger` import |

---

## WU-1: Fix Lemonade message format + extract shared PDF helper

**Issues:** #1 (system-only message), #5 (PDF extraction duplication)
**Files:** `tldr_scholar/backends/lemonade.py`, `tldr_scholar/ingest.py`,
  `tests/test_backends.py`, `tests/test_ingest.py`

### Fix #1: Add user message to Lemonade chat completions

The OpenAI chat completions spec requires at least one user message.
Current code sends only a system message containing both the instruction
and the document. Fix: split into system (instruction) + user (document).

This requires restructuring how the Lemonade backend builds its messages.
Instead of formatting the full `SUMMARY_PROMPT_TEMPLATE` into a single
system message, split it:

```python
def summarize(self, text, max_chars, focus, hashtag_instruction):
    system_msg = (
        f"Summarize the following document in approximately {max_chars} characters.\n"
        f"Focus on: {focus}.\n"
        "Be concise, precise, and factual. Do not add information not in the source.\n"
        f"{hashtag_instruction}"
    ).strip()

    response = httpx.post(
        f"{self._host}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            "stream": False,
        },
        timeout=self._timeout,
    )
```

Note: this means Lemonade no longer uses `SUMMARY_PROMPT_TEMPLATE` — it builds
its own system/user split. The template remains used by Gemini and Ollama
(which use single-prompt APIs, not chat). This is correct: the chat API
naturally separates instruction from content, making `<document>` delimiters
unnecessary for Lemonade.

**Do NOT change Gemini or Ollama** — they use single-prompt APIs where the
template with `<document>` delimiters is the right approach.

### Fix #5: Extract shared PDF extraction helper

Both `_ingest_pdf(path)` and `_ingest_url`'s PDF branch do:
```python
import fitz
import pymupdf4llm
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
pages = list(range(min(20, len(doc))))
text = pymupdf4llm.to_markdown(doc, pages=pages)
```

Extract to a helper that takes an already-opened `fitz.Document` — this keeps
`fitz.open()` in each caller, preserving `_ingest_pdf`'s password detection:

```python
def _pdf_doc_to_text(doc, max_pages: int = 20) -> str:
    """Convert an open fitz.Document to text. Lazy-imports pymupdf4llm."""
    import pymupdf4llm
    pages = list(range(min(max_pages, len(doc))))
    try:
        text = pymupdf4llm.to_markdown(doc, pages=pages)
    except Exception:
        return ""
    return text.strip() if text else ""
```

`_ingest_pdf` opens the doc (with password detection), then calls `_pdf_doc_to_text(doc)`.
`_ingest_url` opens the doc (no password check needed for URLs), then calls `_pdf_doc_to_text(doc)`.
Each caller still does `import fitz; doc = fitz.open(...)` — the shared helper only
deduplicates the `pymupdf4llm.to_markdown` conversion.

### TDD Tests
- Lemonade backend mock: verify `messages` list has both system and user roles
- Existing Lemonade tests still pass (response parsing unchanged)
- Existing ingest tests still pass (behavior unchanged)

---

## WU-2: Remove all unused imports + move urlparse to module top

**Issues:** #2, #3, #4, #6, #7, #8
**Files:** `tldr_scholar/ingest.py`, `tldr_scholar/cli.py`, `tldr_scholar/config.py`,
  `tldr_scholar/hashtags.py`, `tldr_scholar/backends/ollama.py`

### Changes

1. **`ingest.py`**: Remove `from typing import Optional` (unused)

2. **`cli.py`**: Remove `from tldr_scholar.models import SummaryRequest` (unused).
   Move `from urllib.parse import urlparse` to module top (currently lazy inside function).

3. **`config.py`**: Remove `from typing import Any, Optional` (unused)

4. **`hashtags.py`**: Remove `from typing import Optional` (unused)

5. **`backends/ollama.py`**: Add `logger.debug` to the except block rather than
   removing the import — consistency with other backends that log on failure:
   ```python
   except Exception as e:
       logger.debug(f"Ollama request failed: {e}")
       return None
   ```

### TDD Tests
- No behavioral changes — all 64 existing tests pass unchanged

---

## Execution Order

WU-1 (Lemonade + PDF helper) → WU-2 (imports). Sequential.

## Files Modified

| File | WU |
|---|---|
| `tldr_scholar/backends/lemonade.py` | WU-1 |
| `tldr_scholar/ingest.py` | WU-1, WU-2 |
| `tldr_scholar/cli.py` | WU-2 |
| `tldr_scholar/config.py` | WU-2 |
| `tldr_scholar/hashtags.py` | WU-2 |
| `tldr_scholar/backends/ollama.py` | WU-2 |
| `tests/test_backends.py` | WU-1 |

## Success Criteria

1. All 64 existing tests pass
2. Lemonade sends both system + user messages
3. PDF extraction in exactly one function (no duplication)
4. Zero unused imports across all source files
5. `urlparse` imported at module top in cli.py
