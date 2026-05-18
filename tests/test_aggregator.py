"""Tests for tldr_scholar.aggregator (WU-4).

Covers:
- aggregate_topic: 1 LLM call asserted by mock call count
- aggregate_topic: posts field populated when LLM succeeds
- aggregate_topic: posts field STILL populated when LLM fails (warn envelope emitted)
- aggregate_topic: TopicAggregationResult JSON parses + maps to TopicProfile
- aggregate_global: re-uses DEEP_SYNTHESIS_PROMPT (verified by mock inspecting prompt arg)
- aggregate_global: returns dict with agenda/worldview/pivot_logic/identifiable_nuances keys
- End-to-end save/load round-trip: write Persona YAML with populated topics+posts,
  load via PersonaManager, verify posts field byte-identical
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from tldr_scholar.aggregator import (
    TopicAggregationResult,
    aggregate_global,
    aggregate_topic,
)
from tldr_scholar.personas import DeltaRecord, Persona, PersonaManager, TopicProfile
from tldr_scholar.prompts import DEEP_SYNTHESIS_PROMPT, TOPIC_AGGREGATION_PROMPT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_delta(baseline_type: str = "claims") -> DeltaRecord:
    return DeltaRecord(
        baseline_type=baseline_type,  # type: ignore[arg-type]
        statements=["AI reduces costs by 30%."],
        status_per_statement=["shared"],
        intent="Positive framing of AI cost reduction.",
    )


def _valid_aggregation_json(overrides: dict | None = None) -> str:
    data = {
        "revelation_priorities": ["AI reduces costs", "open-source wins"],
        "suppression_rules": ["risks ignored", "bias downplayed"],
        "substantive_anchors": ["cost reduction reframed as efficiency"],
        "rhetorical_strategy": "Techno-optimist framing emphasising economic gains.",
        "confidence": {
            "revelation_priorities": 85,
            "suppression_rules": 70,
            "substantive_anchors": 60,
            "rhetorical_strategy": 80,
        },
    }
    if overrides:
        data.update(overrides)
    return json.dumps(data)


def _valid_global_yaml() -> str:
    """YAML matching DEEP_SYNTHESIS_PROMPT expected output shape."""
    return yaml.dump(
        {
            "profile": {
                "agenda": "Promote open-source AI adoption.",
                "worldview": "Techno-optimist.",
                "pivot_logic": "Reframe risks as manageable.",
                "identifiable_nuances": ["heavy use of passive voice", "citation-dropping"],
            },
            "confidence": {"agenda": 80, "worldview": 75},
        }
    )


# ---------------------------------------------------------------------------
# aggregate_topic — 1 LLM call per topic
# ---------------------------------------------------------------------------


class TestAggregateTopicCallCount:
    """Exactly 1 LLM call is made per aggregate_topic invocation."""

    @pytest.mark.asyncio
    async def test_single_llm_call(self):
        llm_mock = AsyncMock(return_value=_valid_aggregation_json())
        await aggregate_topic(
            label="ai+economics",
            centroid=[0.1, 0.2],
            posts=["AI post 1", "AI post 2"],
            deltas=[_make_delta()],
            llm_call=llm_mock,
        )
        assert llm_mock.call_count == 1


# ---------------------------------------------------------------------------
# aggregate_topic — posts field always populated
# ---------------------------------------------------------------------------


class TestAggregateTopicPostsField:
    """TopicProfile.posts contains the full clustered post text strings."""

    @pytest.mark.asyncio
    async def test_posts_populated_on_success(self):
        posts = ["post A", "post B", "post C"]
        llm_mock = AsyncMock(return_value=_valid_aggregation_json())
        tp = await aggregate_topic(
            label="tech",
            centroid=[0.0] * 4,
            posts=posts,
            deltas=[_make_delta()],
            llm_call=llm_mock,
        )
        assert tp.posts == posts

    @pytest.mark.asyncio
    async def test_posts_populated_when_llm_raises(self):
        posts = ["post X", "post Y"]
        llm_mock = AsyncMock(side_effect=RuntimeError("network error"))

        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            tp = await aggregate_topic(
                label="tech",
                centroid=[0.0],
                posts=posts,
                deltas=[_make_delta()],
                llm_call=llm_mock,
            )

        assert tp.posts == posts, "posts must survive LLM failure"
        # Check warn envelope was emitted
        output = stderr_capture.getvalue()
        envelope = json.loads(output.strip().split("\n")[0])
        assert envelope["code"] == "topic_aggregation_failed"

    @pytest.mark.asyncio
    async def test_posts_populated_when_llm_returns_bad_json(self):
        posts = ["post Z"]
        llm_mock = AsyncMock(return_value="not valid json at all {{{")

        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            tp = await aggregate_topic(
                label="misc",
                centroid=[],
                posts=posts,
                deltas=[_make_delta()],
                llm_call=llm_mock,
            )

        assert tp.posts == posts
        # LLM-derived fields must be empty
        assert tp.revelation_priorities == []
        assert tp.suppression_rules == []
        assert tp.rhetorical_strategy == ""


# ---------------------------------------------------------------------------
# aggregate_topic — warn envelope on failure
# ---------------------------------------------------------------------------


class TestAggregateTopicWarnEnvelope:
    """Emit a warn envelope with code=topic_aggregation_failed on any LLM failure."""

    @pytest.mark.asyncio
    async def test_warn_envelope_emitted_on_llm_exception(self):
        llm_mock = AsyncMock(side_effect=ValueError("LLM exploded"))
        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            await aggregate_topic(
                label="politics",
                centroid=[],
                posts=["p1"],
                deltas=[_make_delta()],
                llm_call=llm_mock,
            )
        raw = stderr_capture.getvalue().strip()
        assert raw, "Envelope must be written to stderr"
        envelope = json.loads(raw.split("\n")[0])
        assert envelope["level"] == "warn"
        assert envelope["code"] == "topic_aggregation_failed"
        assert envelope["stage"] == "aggregator"

    @pytest.mark.asyncio
    async def test_warn_envelope_emitted_on_parse_failure(self):
        llm_mock = AsyncMock(return_value="---\nnot: json")
        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            await aggregate_topic(
                label="environment",
                centroid=[],
                posts=["post"],
                deltas=[_make_delta()],
                llm_call=llm_mock,
            )
        raw = stderr_capture.getvalue().strip()
        assert raw
        envelope = json.loads(raw.split("\n")[0])
        assert envelope["code"] == "topic_aggregation_failed"


# ---------------------------------------------------------------------------
# aggregate_topic — JSON parsing and TopicProfile construction
# ---------------------------------------------------------------------------


class TestAggregateTopicResultMapping:
    """TopicAggregationResult JSON response maps correctly to TopicProfile fields."""

    @pytest.mark.asyncio
    async def test_result_mapped_to_topic_profile(self):
        posts = ["tech post 1"]
        llm_mock = AsyncMock(return_value=_valid_aggregation_json())
        tp = await aggregate_topic(
            label="technology",
            centroid=[1.0, 2.0],
            posts=posts,
            deltas=[_make_delta()],
            llm_call=llm_mock,
        )
        assert isinstance(tp, TopicProfile)
        assert tp.label == "technology"
        assert tp.centroid == [1.0, 2.0]
        assert tp.sample_size == 1
        assert "AI reduces costs" in tp.revelation_priorities
        assert "risks ignored" in tp.suppression_rules
        assert "cost reduction reframed as efficiency" in tp.substantive_anchors
        assert "Techno-optimist" in tp.rhetorical_strategy
        assert "revelation_priorities" in tp.confidence

    @pytest.mark.asyncio
    async def test_markdown_fence_stripped_before_parse(self):
        """JSON wrapped in ```json...``` fences is parsed correctly."""
        fenced = "```json\n" + _valid_aggregation_json() + "\n```"
        llm_mock = AsyncMock(return_value=fenced)
        tp = await aggregate_topic(
            label="tech",
            centroid=[],
            posts=["post"],
            deltas=[_make_delta()],
            llm_call=llm_mock,
        )
        assert tp.revelation_priorities  # non-empty means parse succeeded

    def test_topic_aggregation_result_validates_empty(self):
        """TopicAggregationResult accepts all-default (empty) payload."""
        r = TopicAggregationResult.model_validate({})
        assert r.revelation_priorities == []
        assert r.rhetorical_strategy == ""

    def test_topic_aggregation_result_validates_full(self):
        data = json.loads(_valid_aggregation_json())
        r = TopicAggregationResult.model_validate(data)
        assert r.revelation_priorities == ["AI reduces costs", "open-source wins"]
        assert r.confidence["revelation_priorities"] == 85


# ---------------------------------------------------------------------------
# aggregate_topic — no deltas
# ---------------------------------------------------------------------------


class TestAggregateTopicNoDeltas:
    """If deltas is empty, no LLM call is made and base TopicProfile is returned."""

    @pytest.mark.asyncio
    async def test_no_llm_call_when_no_deltas(self):
        llm_mock = AsyncMock()
        tp = await aggregate_topic(
            label="empty_topic",
            centroid=[0.5],
            posts=["some post"],
            deltas=[],
            llm_call=llm_mock,
        )
        assert llm_mock.call_count == 0
        assert tp.posts == ["some post"]
        assert tp.label == "empty_topic"


# ---------------------------------------------------------------------------
# aggregate_global — uses DEEP_SYNTHESIS_PROMPT
# ---------------------------------------------------------------------------


class TestAggregateGlobal:
    """aggregate_global re-uses DEEP_SYNTHESIS_PROMPT and returns correct keys."""

    @pytest.mark.asyncio
    async def test_uses_deep_synthesis_prompt(self):
        """LLM call receives a prompt that contains DEEP_SYNTHESIS_PROMPT structure."""
        captured_prompts: list[str] = []

        async def capture(prompt: str) -> str:
            captured_prompts.append(prompt)
            return _valid_global_yaml()

        await aggregate_global([_make_delta()], capture)

        assert captured_prompts, "LLM must be called"
        prompt_text = captured_prompts[0]
        # DEEP_SYNTHESIS_PROMPT starts with 'Synthesize a global persona profile'
        assert "Synthesize a global persona profile" in prompt_text
        assert "Delta Reports:" in prompt_text

    @pytest.mark.asyncio
    async def test_returns_required_keys(self):
        llm_mock = AsyncMock(return_value=_valid_global_yaml())
        result = await aggregate_global([_make_delta()], llm_mock)
        assert set(result.keys()) >= {"agenda", "worldview", "pivot_logic", "identifiable_nuances"}

    @pytest.mark.asyncio
    async def test_returns_correct_values(self):
        llm_mock = AsyncMock(return_value=_valid_global_yaml())
        result = await aggregate_global([_make_delta()], llm_mock)
        assert result["agenda"] == "Promote open-source AI adoption."
        assert result["worldview"] == "Techno-optimist."
        assert "heavy use of passive voice" in result["identifiable_nuances"]

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_empty_deltas(self):
        llm_mock = AsyncMock()
        result = await aggregate_global([], llm_mock)
        assert result == {}
        llm_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_llm_failure(self):
        llm_mock = AsyncMock(side_effect=RuntimeError("timeout"))
        result = await aggregate_global([_make_delta()], llm_mock)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_pydantic_tags_in_prompt(self):
        """DeltaRecord must be serialised via model_dump() — no !!python/ YAML tags."""
        captured: list[str] = []

        async def capture(prompt: str) -> str:
            captured.append(prompt)
            return _valid_global_yaml()

        await aggregate_global([_make_delta()], capture)
        assert "!!python/" not in captured[0], "model_dump() must strip Pydantic YAML tags"


# ---------------------------------------------------------------------------
# End-to-end round-trip: write Persona YAML, load via PersonaManager
# ---------------------------------------------------------------------------


class TestPersonaRoundTrip:
    """Write Persona YAML with populated topics+posts; load via PersonaManager; verify posts."""

    def test_posts_field_preserved_round_trip(self):
        posts_for_topic = ["The quick brown fox.", "Jumped over the lazy dog."]
        tp = TopicProfile(
            label="animals",
            centroid=[0.1, 0.2, 0.3],
            sample_size=2,
            posts=posts_for_topic,
            revelation_priorities=["foxes are nimble"],
            suppression_rules=[],
            substantive_anchors=[],
            rhetorical_strategy="Simple factual prose.",
            confidence={"revelation_priorities": 90},
        )
        persona = Persona(
            name="round_trip_test",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            status="complete",
            topics={"animals": tp},
            agenda="Test agenda.",
            worldview="Test worldview.",
            pivot_logic="",
            identifiable_nuances=["marker1"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "round_trip_test.yaml"
            with open(output_path, "w") as f:
                yaml.dump(persona.model_dump(), f, sort_keys=False)

            manager = PersonaManager(config_dir=Path(tmpdir))
            manager.reload()
            loaded = manager.get_persona("round_trip_test")

        assert loaded is not None, "Persona must be loadable after YAML write"
        assert "animals" in loaded.topics
        loaded_tp = loaded.topics["animals"]
        assert loaded_tp.posts == posts_for_topic, (
            f"posts field must be byte-identical after round-trip; "
            f"got {loaded_tp.posts!r}"
        )

    def test_multiple_topics_all_posts_preserved(self):
        topics = {
            "tech": TopicProfile(
                label="tech",
                centroid=[0.5],
                sample_size=1,
                posts=["AI is transformative."],
                revelation_priorities=["AI adoption"],
            ),
            "policy": TopicProfile(
                label="policy",
                centroid=[0.3],
                sample_size=2,
                posts=["Regulation needed.", "Safety first."],
            ),
        }
        persona = Persona(
            name="multi_topic_test",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            status="complete",
            topics=topics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "multi_topic_test.yaml"
            with open(output_path, "w") as f:
                yaml.dump(persona.model_dump(), f, sort_keys=False)

            manager = PersonaManager(config_dir=Path(tmpdir))
            manager.reload()
            loaded = manager.get_persona("multi_topic_test")

        assert loaded is not None
        assert loaded.topics["tech"].posts == ["AI is transformative."]
        assert loaded.topics["policy"].posts == ["Regulation needed.", "Safety first."]


# ---------------------------------------------------------------------------
# TOPIC_AGGREGATION_PROMPT — content invariants
# ---------------------------------------------------------------------------


class TestTopicAggregationPrompt:
    """TOPIC_AGGREGATION_PROMPT wraps input in <untrusted_content> delimiters."""

    def test_has_untrusted_content_delimiters(self):
        assert "<untrusted_content>" in TOPIC_AGGREGATION_PROMPT
        assert "</untrusted_content>" in TOPIC_AGGREGATION_PROMPT

    def test_references_delta_records_json_placeholder(self):
        assert "{delta_records_json}" in TOPIC_AGGREGATION_PROMPT

    def test_requests_json_output(self):
        prompt_lower = TOPIC_AGGREGATION_PROMPT.lower()
        assert "json" in prompt_lower

    def test_lists_all_five_fields(self):
        for field in [
            "revelation_priorities",
            "suppression_rules",
            "substantive_anchors",
            "rhetorical_strategy",
            "confidence",
        ]:
            assert field in TOPIC_AGGREGATION_PROMPT
