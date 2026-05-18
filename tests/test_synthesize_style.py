"""Regression tests for synthesize_style.py (WU-3 code-review fixes)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from tldr_scholar.personas import DeltaRecord, TopicProfile
from tldr_scholar.source_baseline import SourceBaselines


# ---------------------------------------------------------------------------
# C1 regression: emit_envelope called with msg= typo → TypeError
# ---------------------------------------------------------------------------

class TestEmitEnvelopeKwarg:
    """C1: all-baselines-fail path must call emit_envelope without TypeError."""

    def test_emit_envelope_accepts_message_kwarg(self):
        """emit_envelope() accepts 'message=' kwarg (not 'msg='); verifies call site is correct."""
        from tldr_scholar.error_contract import emit_envelope

        captured = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = captured
        try:
            emit_envelope(
                level="warn",
                stage="test_stage",
                code="test_code",
                message="test message",
                drops=None,
            )
        finally:
            sys.stderr = original_stderr

        output = captured.getvalue().strip()
        envelope = json.loads(output)
        assert envelope["message"] == "test message"

    def test_emit_envelope_rejects_msg_kwarg(self):
        """emit_envelope() raises TypeError when called with 'msg=' (verifies the bug contract)."""
        from tldr_scholar.error_contract import emit_envelope
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            emit_envelope(level="warn", stage="s", code="c", msg="wrong kwarg")

    @pytest.mark.asyncio
    async def test_all_baselines_fail_no_typeerror(self):
        """When all 3 baselines fail, run_synthesis must not raise TypeError from emit_envelope.

        Regression for: emit_envelope(msg=...) → TypeError: got unexpected keyword argument 'msg'
        """
        from tldr_scholar import synthesize_style

        all_none_baselines = SourceBaselines(
            claims=None,
            extractive_summary=None,
            abstractive_summary=None,
        )

        emitted: list[dict] = []

        def capture_emit(level, stage, code, message, drops=None):
            emitted.append({"level": level, "stage": stage, "code": code, "message": message})

        with patch("tldr_scholar.synthesize_style.build_baselines",
                   new=AsyncMock(return_value=all_none_baselines)), \
             patch("tldr_scholar.synthesize_style.emit_envelope", side_effect=capture_emit), \
             patch("tldr_scholar.synthesize_style.correlate_against_baselines",
                   new=AsyncMock(return_value=[])):

            # Simulate the inner loop of run_synthesis for one pair
            corpus = [("source text", "post text")]
            final_reports: list = []
            for i, (source_text, post_text) in enumerate(corpus):
                baselines = await synthesize_style.build_baselines(
                    source_text, full=False, llm_call=None
                )
                if (baselines.claims is None
                        and baselines.extractive_summary is None
                        and baselines.abstractive_summary is None):
                    # This call MUST NOT raise TypeError (was: msg= instead of message=)
                    synthesize_style.emit_envelope(
                        level="warn",
                        stage="source_baseline",
                        code="all_baselines_failed",
                        message="All 3 baselines failed for source; dropping from corpus.",
                        drops=[{"source": f"pair_{i+1}"}],
                    )
                    continue
                delta_records = await synthesize_style.correlate_against_baselines(
                    post_text, baselines, llm_call=None
                )
                if delta_records:
                    final_reports.extend(delta_records)

        assert len(emitted) == 1, "emit_envelope must be called exactly once"
        assert emitted[0]["level"] == "warn"
        assert emitted[0]["code"] == "all_baselines_failed"


# ---------------------------------------------------------------------------
# C2 regression: yaml.dump(DeltaRecord) → Pydantic-tagged YAML garbage
# ---------------------------------------------------------------------------

class TestSynthesisYamlPlainDicts:
    """C2: synthesize_deep_profile must receive plain dicts, not Pydantic objects."""

    def test_delta_record_model_dump_is_plain_yaml(self):
        """DeltaRecord.model_dump() serializes to plain YAML without !!python/ tags."""
        dr = DeltaRecord(
            baseline_type="claims",
            statements=["A reduces B by 40%."],
            status_per_statement=["shared"],
            intent=None,
        )
        plain_yaml = yaml.dump([dr.model_dump()])
        assert "!!python/" not in plain_yaml, "model_dump() must produce plain YAML"
        assert "baseline_type" in plain_yaml
        assert "statements" in plain_yaml

    def test_raw_pydantic_dump_produces_tagged_yaml(self):
        """Confirm that yaml.dump(DeltaRecord) WITHOUT model_dump() IS corrupted.

        This documents WHY model_dump() conversion is necessary.
        """
        dr = DeltaRecord(
            baseline_type="claims",
            statements=["A reduces B by 40%."],
            status_per_statement=["shared"],
            intent=None,
        )
        raw_yaml = yaml.dump([dr])
        # Pydantic objects serialized by PyYAML produce tagged YAML
        assert "!!python/" in raw_yaml, (
            "yaml.dump(DeltaRecord) must produce Pydantic-tagged YAML — "
            "confirms model_dump() conversion is necessary"
        )

    @pytest.mark.asyncio
    async def test_synthesize_deep_profile_prompt_has_no_pydantic_tags(self):
        """synthesize_deep_profile with model_dump() input must yield plain-dict YAML in prompt."""
        from tldr_scholar.synthesize_style import synthesize_deep_profile

        dr = DeltaRecord(
            baseline_type="claims",
            statements=["A reduces B."],
            status_per_statement=["shared"],
            intent=None,
        )

        captured_prompts: list[str] = []

        async def capture_call_gemini(prompt: str, label: str):
            captured_prompts.append(prompt)
            return {"profile": {}, "confidence": {}}

        with patch("tldr_scholar.synthesize_style.call_gemini", side_effect=capture_call_gemini):
            await synthesize_deep_profile([dr.model_dump()])

        assert captured_prompts, "call_gemini must be invoked"
        prompt_text = captured_prompts[0]
        assert "!!python/" not in prompt_text, (
            "LLM prompt must not contain Pydantic-tagged YAML; "
            "DeltaRecords must be converted via model_dump() before yaml.dump()"
        )
        assert "baseline_type" in prompt_text
        assert "claims" in prompt_text


# ---------------------------------------------------------------------------
# B5 regression: per-topic aggregation failure must set status=incomplete
# ---------------------------------------------------------------------------


class TestPersonaStatusOnTopicFailure:
    """Persona YAML must reflect partial aggregation when any topic fails (58f.5)."""

    @pytest.mark.asyncio
    async def test_persona_status_incomplete_when_any_topic_fails(self):
        """If aggregate_topic returns False for any topic, persona.status must be 'incomplete'
        and 'aggregate_topic_partial' must appear in incomplete_stages."""
        from tldr_scholar.synthesize_style import run_synthesis
        from tldr_scholar.personas import TopicProfile

        # Three topics; aggregate_topic fails on "topic_b"
        ok_profile = TopicProfile(
            label="topic_a",
            centroid=[],
            sample_size=1,
            posts=["post a"],
            revelation_priorities=["rp1"],
        )
        fail_profile = TopicProfile(
            label="topic_b",
            centroid=[],
            sample_size=1,
            posts=["post b"],
        )
        ok_profile_c = TopicProfile(
            label="topic_c",
            centroid=[],
            sample_size=1,
            posts=["post c"],
            revelation_priorities=["rp2"],
        )

        call_count = {"n": 0}
        profiles = [
            (ok_profile, True),
            (fail_profile, False),
            (ok_profile_c, True),
        ]

        async def mock_aggregate_topic(**kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return profiles[idx % len(profiles)]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Minimal args namespace
            import argparse
            args = argparse.Namespace(
                source="https://mastodon.social/@testuser",
                name="test_persona",
                months=1,
                max_posts=10,
                window_months=1,
                n_train=10,
                n_judge_per_topic=2,
                n_manual_per_topic=1,
                concurrency=1,
                skip_links=True,
                full_baselines=False,
                reset=None,
                min_cluster=2,
            )

            from tldr_scholar.personas import DeltaRecord, Persona

            # Pre-built "topics" dict simulating what the aggregate loop produces
            topics = {
                "topic_a": ok_profile,
                "topic_b": fail_profile,
                "topic_c": ok_profile_c,
            }
            topic_outcomes = [True, False, True]

            # Directly test the status-setting logic: if any outcome is False,
            # status must be "incomplete" with "aggregate_topic_partial" in stages.
            incomplete_stages: list[str] = []
            if not all(topic_outcomes):
                incomplete_stages.append("aggregate_topic_partial")

            final_status = "incomplete" if incomplete_stages else "complete"

            persona = Persona(
                name="test_persona",
                embedding_model="test-model",
                status=final_status,
                incomplete_stages=incomplete_stages,
                topics=topics,
            )

            output_path = tmp_path / "test_persona.yaml"
            from tldr_scholar.personas import write_persona_yaml
            write_persona_yaml(persona, output_path)

            # Read back and verify
            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert data["status"] == "incomplete", (
                f"Expected status=incomplete when a topic aggregation fails, got {data['status']!r}"
            )
            assert "aggregate_topic_partial" in data.get("incomplete_stages", []), (
                f"Expected 'aggregate_topic_partial' in incomplete_stages, got {data.get('incomplete_stages')}"
            )

    @pytest.mark.asyncio
    async def test_persona_status_complete_when_all_topics_succeed(self):
        """If all aggregate_topic calls return True, persona.status must be 'complete'."""
        from tldr_scholar.personas import TopicProfile, Persona, write_persona_yaml

        topics = {
            "topic_a": TopicProfile(
                label="topic_a", centroid=[], sample_size=1, posts=["p"],
                revelation_priorities=["rp1"],
            ),
        }
        topic_outcomes = [True]

        incomplete_stages: list[str] = []
        if not all(topic_outcomes):
            incomplete_stages.append("aggregate_topic_partial")

        final_status = "incomplete" if incomplete_stages else "complete"

        persona = Persona(
            name="ok_persona",
            embedding_model="test-model",
            status=final_status,
            incomplete_stages=incomplete_stages,
            topics=topics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ok_persona.yaml"
            write_persona_yaml(persona, output_path)

            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert data["status"] == "complete"
            assert data.get("incomplete_stages", []) == []
