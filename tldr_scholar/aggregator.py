"""Per-topic and global aggregation for the NWL persona pipeline (WU-4).

aggregate_topic : 1 Gemini call per topic cluster → tuple[TopicProfile, bool]
aggregate_global: re-uses DEEP_SYNTHESIS_PROMPT over all DeltaRecords → top-level Persona fields
"""
from __future__ import annotations

import json
import logging
import yaml
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from tldr_scholar.error_contract import emit_envelope as emit
from tldr_scholar.personas import DeltaRecord, TopicProfile
from tldr_scholar.prompts import DEEP_SYNTHESIS_PROMPT, TOPIC_AGGREGATION_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
LLMCaller = Callable[[str], Awaitable[str]]

# ---------------------------------------------------------------------------
# Pydantic sub-schema for LLM aggregation result
# ---------------------------------------------------------------------------


class TopicAggregationResult(BaseModel):
    """Five fields returned by TOPIC_AGGREGATION_PROMPT."""

    revelation_priorities: list[str] = []
    suppression_rules: list[str] = []
    substantive_anchors: list[str] = []
    rhetorical_strategy: str = ""
    confidence: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Per-topic aggregation
# ---------------------------------------------------------------------------


async def aggregate_topic(
    label: str,
    centroid: list[float],
    posts: list[str],
    deltas: list[DeltaRecord],
    llm_call: LLMCaller,
) -> tuple[TopicProfile, bool]:
    """Build one TopicProfile for *label*.

    The ``posts`` field is always populated from the *posts* argument,
    independent of LLM success.  On LLM failure the five LLM-derived fields
    are empty and a warn envelope is emitted.

    Parameters
    ----------
    label:     Topic key string (e.g. "economics+labor+wage").
    centroid:  384-d embedding centroid for the topic cluster.
    posts:     Full text of every training post assigned to this topic.
    deltas:    DeltaRecord list for (post, source) pairs in this cluster.
    llm_call:  Async LLM callable; injected so tests can mock.

    Returns
    -------
    tuple[TopicProfile, bool]
        TopicProfile with posts always populated; bool is True on LLM success,
        False on LLM failure or JSON parse failure (degraded/fallback result).
    """
    sample_size = len(posts)

    # Base profile — posts always populated regardless of LLM outcome.
    base = dict(
        label=label,
        centroid=centroid,
        sample_size=sample_size,
        posts=list(posts),
    )

    if not deltas:
        # Nothing to aggregate; return base with empty LLM fields (no failure).
        return TopicProfile(**base), True

    delta_records_json = json.dumps(
        [d.model_dump() for d in deltas], ensure_ascii=False
    )
    prompt = TOPIC_AGGREGATION_PROMPT.format(delta_records_json=delta_records_json)

    try:
        raw = await llm_call(prompt)
    except Exception as exc:
        logger.warning("aggregator: topic=%s LLM call failed — %s", label, exc)
        _emit_topic_fail(label, str(exc))
        return TopicProfile(**base), False

    result = _parse_aggregation_result(raw)
    if result is None:
        logger.warning("aggregator: topic=%s JSON parse failed", label)
        _emit_topic_fail(label, "JSON parse failed")
        return TopicProfile(**base), False

    return TopicProfile(
        **base,
        revelation_priorities=result.revelation_priorities,
        suppression_rules=result.suppression_rules,
        substantive_anchors=result.substantive_anchors,
        rhetorical_strategy=result.rhetorical_strategy,
        confidence=result.confidence,
    ), True


def _parse_aggregation_result(raw: Any) -> TopicAggregationResult | None:
    """Parse TOPIC_AGGREGATION_PROMPT JSON response into TopicAggregationResult.

    Strips Markdown code fences if present.  Returns None on any parse error.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    # Strip Markdown JSON fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1].strip()
            if text.startswith("json\n"):
                text = text[5:]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return TopicAggregationResult.model_validate(data)
    except ValidationError:
        return None


def _emit_topic_fail(label: str, reason: str) -> None:
    emit(
        level="warn",
        stage="aggregator",
        code="topic_aggregation_failed",
        message="Per-topic LLM aggregation failed; topic will have empty LLM-derived fields.",
        drops=[{"source": label, "reason": reason}],
    )


# ---------------------------------------------------------------------------
# Global aggregation
# ---------------------------------------------------------------------------


async def aggregate_global(
    deltas: list[DeltaRecord],
    llm_call: LLMCaller,
) -> dict[str, Any]:
    """Run DEEP_SYNTHESIS_PROMPT over all DeltaRecords → top-level Persona fields.

    Returns a dict with keys: agenda, worldview, pivot_logic, identifiable_nuances.
    On failure returns an empty dict (caller fills defaults).
    """
    if not deltas:
        return {}

    reports_yaml = yaml.dump([d.model_dump() for d in deltas])
    prompt = DEEP_SYNTHESIS_PROMPT.format(reports=reports_yaml)

    try:
        raw = await llm_call(prompt)
    except Exception as exc:
        logger.warning("aggregator: global synthesis LLM call failed — %s", exc)
        return {}

    parsed = _parse_global_response(raw)
    if parsed is None:
        logger.warning("aggregator: global synthesis parse failed")
        return {}

    return parsed


def _parse_global_response(raw: Any) -> dict[str, Any] | None:
    """Parse DEEP_SYNTHESIS_PROMPT response.

    The prompt asks for a YAML dict with 'profile' and 'confidence' keys.
    Returns the flattened top-level fields relevant to Persona, or None on error.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    # Strip Markdown YAML fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1].strip()
            if text.startswith("yaml\n"):
                text = text[5:]
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None

    profile = data.get("profile", data)
    if not isinstance(profile, dict):
        profile = {}

    return {
        "agenda": str(profile.get("agenda", "")),
        "worldview": str(profile.get("worldview", "")),
        "pivot_logic": str(profile.get("pivot_logic", "")),
        "identifiable_nuances": list(profile.get("identifiable_nuances", [])),
    }
