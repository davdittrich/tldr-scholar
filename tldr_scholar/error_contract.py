"""Error contract for tldr-scholar: envelope emit, exit codes, drop summary.

Replaces the WU-2 _envelope.py stub.  All envelopes go to stderr;
end-of-run drop summary goes to stdout.

Scrubbing rules (enforced on every outbound string):
- Absolute paths (starting with '/') → os.path.basename.
- Environment-variable values present in any string → '<env>'.
- URL query strings in drops[].source → stripped (signed-URL guard).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.parse import urlparse, urlunparse

# ---------------------------------------------------------------------------
# Exit code registry
# ---------------------------------------------------------------------------

EXIT_CODES: dict[str, int] = {
    "success": 0,
    "internal": 1,
    "v2_schema_fail": 2,
    "embedding_mismatch": 3,
    "llm_exhausted": 4,
    "empty_corpus": 5,
}

# ---------------------------------------------------------------------------
# Module-level drop counter (reset between runs / in tests via reset_drop_counter)
# ---------------------------------------------------------------------------

_DROP_COUNTER: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Scrubbing helpers
# ---------------------------------------------------------------------------

def _scrub_string(s: str) -> str:
    """Redact env-var values and replace absolute paths with basenames."""
    # Env-var values
    for v in os.environ.values():
        if v and len(v) > 3 and v in s:
            s = s.replace(v, "<env>")
    # Absolute path → basename
    if s.startswith("/"):
        s = os.path.basename(s) or "<scrubbed-path>"
    return s


def _scrub_source(src: str) -> str:
    """Strip URL query strings (signed URLs / auth tokens), then scrub."""
    try:
        parts = urlparse(src)
        if parts.query or parts.fragment:
            src = urlunparse(parts._replace(query="", fragment=""))
    except Exception:
        pass
    return _scrub_string(src)


# ---------------------------------------------------------------------------
# Envelope emit
# ---------------------------------------------------------------------------

def emit_envelope(
    level: str,
    stage: str,
    code: str,
    message: str,
    drops: list[dict[str, str]] | None = None,
) -> None:
    """Write one JSON envelope line to stderr.

    Args:
        level:   Severity string, e.g. "warn" or "error".
        stage:   Pipeline stage label, e.g. "scrape_filter".
        code:    Machine-readable event code, e.g. "injection_filter_match".
        message: Human-readable description (must not contain matched content).
        drops:   Optional list of {"source": ..., "reason": ...} dicts.
                 source MUST be a URL or opaque identifier — never matched text.
    """
    scrubbed_drops: list[dict[str, Any]] = []
    if drops:
        for d in drops:
            reason = d.get("reason", "<unknown>")
            scrubbed_drops.append(
                {
                    "source": _scrub_source(d.get("source", "")),
                    "reason": reason,
                }
            )
            _DROP_COUNTER[reason] = _DROP_COUNTER.get(reason, 0) + 1

    envelope: dict[str, Any] = {
        "level": level,
        "stage": stage,
        "code": code,
        "message": _scrub_string(message),
        "drops": scrubbed_drops,
    }
    sys.stderr.write(json.dumps(envelope) + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# End-of-run drop summary (stdout)
# ---------------------------------------------------------------------------

def emit_drop_summary() -> None:
    """Write per-reason drop counts to stdout.

    Always called at end of run (success or exit 4).  No-op if no drops
    accumulated.
    """
    if not _DROP_COUNTER:
        return
    sys.stdout.write("=== Drop summary ===\n")
    for reason, count in sorted(_DROP_COUNTER.items()):
        sys.stdout.write(f"  {reason}: {count}\n")
    sys.stdout.flush()


def reset_drop_counter() -> None:
    """Clear the module-level drop counter.  For tests and between runs."""
    _DROP_COUNTER.clear()
