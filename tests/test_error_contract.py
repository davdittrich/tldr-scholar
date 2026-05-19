"""Tests for tldr_scholar.error_contract (WU-5a)."""
from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import pytest
import yaml

from tldr_scholar.error_contract import (
    EXIT_CODES,
    emit_envelope,
    emit_drop_summary,
    reset_drop_counter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_stderr(fn, *args, **kwargs):
    """Call *fn* and return what was written to stderr."""
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stderr = old
    return buf.getvalue()


def _capture_stdout(fn, *args, **kwargs):
    """Call *fn* and return what was written to stdout."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ---------------------------------------------------------------------------
# EXIT_CODES table
# ---------------------------------------------------------------------------

class TestExitCodesTable:
    def test_all_six_entries_present(self):
        assert set(EXIT_CODES) == {
            "success",
            "internal",
            "v2_schema_fail",
            "embedding_mismatch",
            "llm_exhausted",
            "empty_corpus",
        }

    def test_correct_values(self):
        assert EXIT_CODES["success"] == 0
        assert EXIT_CODES["internal"] == 1
        assert EXIT_CODES["v2_schema_fail"] == 2
        assert EXIT_CODES["embedding_mismatch"] == 3
        assert EXIT_CODES["llm_exhausted"] == 4
        assert EXIT_CODES["empty_corpus"] == 5


# ---------------------------------------------------------------------------
# emit_envelope channel separation
# ---------------------------------------------------------------------------

class TestEmitEnvelopeChannels:
    def setup_method(self):
        reset_drop_counter()

    def test_writes_to_stderr(self):
        out = _capture_stderr(emit_envelope, "warn", "stage", "code", "msg text")
        assert out.strip() != "", "emit_envelope must write to stderr"

    def test_does_not_write_to_stdout(self):
        out = _capture_stdout(emit_envelope, "warn", "stage", "code", "msg text")
        assert out == "", "emit_envelope must NOT write to stdout"

    def test_json_parseable(self):
        raw = _capture_stderr(emit_envelope, "warn", "stage", "code", "hello")
        env = json.loads(raw.strip())
        assert env["level"] == "warn"
        assert env["stage"] == "stage"
        assert env["code"] == "code"
        assert env["message"] == "hello"
        assert "drops" in env

    def test_rejects_msg_kwarg(self):
        """emit_envelope uses 'message=' not 'msg='; wrong kwarg must raise TypeError."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            emit_envelope(level="warn", stage="s", code="c", msg="wrong")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Scrubbing: absolute paths
# ---------------------------------------------------------------------------

class TestScrubAbsolutePaths:
    def setup_method(self):
        reset_drop_counter()

    def test_absolute_path_in_message_becomes_basename(self):
        raw = _capture_stderr(emit_envelope, "warn", "s", "c", "/home/user/secret/file.txt")
        env = json.loads(raw.strip())
        assert env["message"] == "file.txt"

    def test_absolute_path_in_drops_source_becomes_basename(self):
        raw = _capture_stderr(
            emit_envelope, "warn", "s", "c", "msg",
            drops=[{"source": "/etc/private/key.pem", "reason": "r"}],
        )
        env = json.loads(raw.strip())
        assert env["drops"][0]["source"] == "key.pem"


# ---------------------------------------------------------------------------
# Scrubbing: env var values
# ---------------------------------------------------------------------------

class TestScrubEnvValues:
    def setup_method(self):
        reset_drop_counter()

    def test_env_var_value_in_message_replaced(self, monkeypatch):
        secret = "SUPERSECRET_12345_VALUE"
        monkeypatch.setenv("_TEST_SECRET_EC", secret)
        raw = _capture_stderr(
            emit_envelope, "warn", "s", "c", f"found secret: {secret}"
        )
        env = json.loads(raw.strip())
        assert secret not in env["message"], "env var value must be scrubbed"
        assert "<env>" in env["message"]

    def test_short_env_values_not_scrubbed(self, monkeypatch):
        """Values with len <= 3 are not scrubbed (avoid false positives)."""
        monkeypatch.setenv("_TEST_SHORT_EC", "XY")
        raw = _capture_stderr(emit_envelope, "warn", "s", "c", "value is XY here")
        env = json.loads(raw.strip())
        assert "XY" in env["message"]


# ---------------------------------------------------------------------------
# Scrubbing: URL query strings in drops[].source
# ---------------------------------------------------------------------------

class TestScrubURLQueryStrings:
    def setup_method(self):
        reset_drop_counter()

    def test_query_string_stripped_from_source(self):
        signed_url = "https://storage.example.com/obj?sig=abc123&token=xyz"
        raw = _capture_stderr(
            emit_envelope, "warn", "s", "c", "msg",
            drops=[{"source": signed_url, "reason": "r"}],
        )
        env = json.loads(raw.strip())
        src = env["drops"][0]["source"]
        assert "?" not in src, "query string must be stripped from drops[].source"
        assert "sig=" not in src
        assert "storage.example.com" in src

    def test_clean_url_unchanged(self):
        clean_url = "https://example.com/path"
        raw = _capture_stderr(
            emit_envelope, "warn", "s", "c", "msg",
            drops=[{"source": clean_url, "reason": "r"}],
        )
        env = json.loads(raw.strip())
        assert env["drops"][0]["source"] == clean_url


# ---------------------------------------------------------------------------
# Content echo policy: matched injection content not echoed
# ---------------------------------------------------------------------------

class TestNoContentEcho:
    def setup_method(self):
        reset_drop_counter()

    def test_injection_text_not_in_message(self, monkeypatch):
        """Content placed in message field is subject to scrubbing only — it should
        not be the actual matched post text.  We verify the env var scrub applies
        (simulating the convention that matched text is never passed as message)."""
        secret = "INJECTION_PAYLOAD_CONTENT_9876"
        monkeypatch.setenv("_TEST_INJECT_EC", secret)
        raw = _capture_stderr(
            emit_envelope, "warn", "s", "injection_filter_match",
            f"dropped source with token {secret}",
        )
        env = json.loads(raw.strip())
        assert secret not in env["message"]

    def test_drops_reason_is_scrubbed(self, monkeypatch):
        """drops[].reason must be passed through _scrub_string — injection-content
        must not appear verbatim in the emitted envelope."""
        secret = "DROPS_REASON_INJECTION_XYZ"
        monkeypatch.setenv("_TEST_INJECT_REASON", secret)
        raw = _capture_stderr(
            emit_envelope, "warn", "s", "c", "msg",
            drops=[{"source": "test_source", "reason": secret}],
        )
        env = json.loads(raw.strip())
        assert env["drops"][0]["reason"] != secret, (
            "drops[].reason must be scrubbed but was echoed verbatim"
        )


# ---------------------------------------------------------------------------
# Drop summary: stdout, not stderr
# ---------------------------------------------------------------------------

class TestDropSummary:
    def setup_method(self):
        reset_drop_counter()

    def test_summary_written_to_stdout_not_stderr(self):
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "x", "reason": "r1"}])
        stderr_out = _capture_stderr(emit_drop_summary)
        stdout_out = _capture_stdout(emit_drop_summary)
        assert stderr_out == "", "drop summary must NOT go to stderr"
        assert "r1" in stdout_out

    def test_summary_lists_reason_count(self):
        reset_drop_counter()
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "a", "reason": "rA"}])
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "b", "reason": "rA"}])
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "c", "reason": "rB"}])
        out = _capture_stdout(emit_drop_summary)
        assert "rA: 2" in out
        assert "rB: 1" in out

    def test_no_output_when_no_drops(self):
        reset_drop_counter()
        out = _capture_stdout(emit_drop_summary)
        assert out == ""


# ---------------------------------------------------------------------------
# Drop counter: accumulates + reset
# ---------------------------------------------------------------------------

class TestDropCounter:
    def setup_method(self):
        reset_drop_counter()

    def test_accumulates_across_calls(self):
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "x", "reason": "reason_x"}])
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "y", "reason": "reason_x"}])
        from tldr_scholar.error_contract import _DROP_COUNTER
        assert _DROP_COUNTER.get("reason_x", 0) == 2

    def test_reset_clears_counter(self):
        emit_envelope("warn", "s", "c", "m", drops=[{"source": "x", "reason": "r"}])
        reset_drop_counter()
        from tldr_scholar.error_contract import _DROP_COUNTER
        assert _DROP_COUNTER == {}


# ---------------------------------------------------------------------------
# PersonaManager: warn on incomplete persona load
# ---------------------------------------------------------------------------

class TestPersonaManagerIncompleteWarn:
    def setup_method(self):
        reset_drop_counter()

    def test_get_persona_emits_warn_for_incomplete(self, tmp_path):
        """PersonaManager.get_persona() emits warn envelope when persona status=incomplete."""
        from tldr_scholar.personas import PersonaManager

        # Write a minimal v2 incomplete persona YAML
        persona_data = {
            "name": "incomplete_persona",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "status": "incomplete",
            "incomplete_stages": ["aggregate_topic"],
            "topics": {
                "_global": {
                    "label": "_global",
                    "centroid": [0.0] * 10,
                    "sample_size": 0,
                }
            },
        }
        (tmp_path / "incomplete_persona.yaml").write_text(yaml.safe_dump(persona_data))

        mgr = PersonaManager(config_dir=tmp_path)

        stderr_buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = stderr_buf
        try:
            persona = mgr.get_persona("incomplete_persona")
        finally:
            sys.stderr = old_stderr

        assert persona is not None
        lines = [l.strip() for l in stderr_buf.getvalue().splitlines() if l.strip()]
        assert len(lines) >= 1, "Expected at least one JSON envelope on stderr"
        env = json.loads(lines[0])
        assert env["level"] == "warn"
        assert env["code"] == "persona_incomplete"

    def test_get_persona_no_warn_for_complete(self, tmp_path):
        """No warn envelope when persona status=complete."""
        from tldr_scholar.personas import PersonaManager

        persona_data = {
            "name": "ok_persona",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "status": "complete",
            "topics": {
                "_global": {
                    "label": "_global",
                    "centroid": [0.0] * 10,
                    "sample_size": 0,
                }
            },
        }
        (tmp_path / "ok_persona.yaml").write_text(yaml.safe_dump(persona_data))

        mgr = PersonaManager(config_dir=tmp_path)

        stderr_buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = stderr_buf
        try:
            mgr.get_persona("ok_persona")
        finally:
            sys.stderr = old_stderr

        output = stderr_buf.getvalue().strip()
        assert output == "", f"No envelope expected for complete persona, got: {output!r}"
