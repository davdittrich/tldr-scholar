"""Minimal emit-envelope stub for tldr-scholar.

Writes a single JSON line to stderr.  WU-5a will replace this
implementation; call sites (corpus_sampler) remain unchanged.

Scrubbing rules (per spec error-contract section):
- drops[].source accepts only a URL or opaque ID string; matched content is
  NEVER included.
- Absolute paths (starting with '/') are replaced with their basename.
- Environment-variable values present in any string field are replaced with
  '<redacted>'.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any


def _scrub(value: str) -> str:
    """Replace env-var values and absolute paths with safe tokens."""
    for v in os.environ.values():
        if v and len(v) > 3 and v in value:
            value = value.replace(v, "<redacted>")
    # Absolute path heuristic: replace leading /…/…/ segments with basename
    if value.startswith("/") and "/" in value[1:]:
        value = value.split("/")[-1]
    return value


def emit(
    level: str,
    stage: str,
    code: str,
    message: str,
    drops: list[dict[str, str]] | None = None,
) -> None:
    """Write a JSON envelope line to stderr.

    Args:
        level:   Severity string, e.g. "warn" or "error".
        stage:   Pipeline stage label, e.g. "scrape_filter" or "corpus_sampler".
        code:    Machine-readable event code, e.g. "injection_filter_match".
        message: Human-readable description (must not contain matched content).
        drops:   Optional list of {source, reason} dicts.  'source' MUST be a
                 URL or opaque id — never the matched text content.
    """
    envelope: dict[str, Any] = {
        "level": level,
        "stage": stage,
        "code": code,
        "message": _scrub(message),
    }
    if drops is not None:
        envelope["drops"] = [
            {"source": _scrub(d.get("source", "")), "reason": d.get("reason", "")}
            for d in drops
        ]
    sys.stderr.write(json.dumps(envelope) + "\n")
    sys.stderr.flush()
