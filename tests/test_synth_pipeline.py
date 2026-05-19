"""Smoke test for the run_synthesis() end-to-end pipeline (v2 CLI)."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldr_scholar.personas import DeltaRecord, TopicProfile
from tldr_scholar.scrapers import SocialPost
from tldr_scholar.source_baseline import SourceBaselines


# ---------------------------------------------------------------------------
# Minimal fixture data
# ---------------------------------------------------------------------------

_POST = SocialPost(
    text="Test post body.",
    timestamp=datetime(2025, 1, 1),
    source_url="https://bsky.app/profile/example.bsky.social",
)

_BASELINES = SourceBaselines(
    claims=["A implies B."],
    extractive_summary=None,
    abstractive_summary=None,
)

_DELTA = DeltaRecord(
    baseline_type="claims",
    statements=["A implies B."],
    status_per_statement=["shared"],
    intent="example",
)

_TOPIC_PROFILE = TopicProfile(
    label="t1",
    centroid=[0.1, 0.2],
    sample_size=1,
    posts=[_POST.text],
)

_GLOBAL_FIELDS = {
    "agenda": "test agenda",
    "worldview": "test worldview",
    "pivot_logic": "test pivot",
    "identifiable_nuances": ["nuance A"],
}


# ---------------------------------------------------------------------------
# Smoke: run_synthesis() produces persona YAML at DEFAULT_PERSONA_DIR/<name>.yaml
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_synthesis_emits_persona_yaml(tmp_path: Path) -> None:
    """run_synthesis() must write <name>.yaml into DEFAULT_PERSONA_DIR.

    Mocks all I/O boundaries (scraper, LLM, cache) so the test stays offline
    and fast.  Replaces the stale test_jsonl_pipeline which exercised the
    removed --format/--output flags.
    """
    from tldr_scholar.synthesize_style import run_synthesis

    args = argparse.Namespace(
        source="https://bsky.app/profile/example.bsky.social",
        name="smoke-persona",
        window_months=1,
        n_train=10,
        n_judge_per_topic=1,
        n_manual_per_topic=1,
        concurrency=1,
        skip_links=True,
        full_baselines=False,
        reset=None,
        min_cluster=2,
    )

    mock_corpus_result = {
        "training": [_POST],
        "training_topic_labels": ["t1"],
        "topic_centroids": {"t1": [0.1, 0.2]},
    }

    with (
        patch("tldr_scholar.synthesize_style.check_embedding_model_cached"),
        patch("tldr_scholar.synthesize_style.ACP_AVAILABLE", True),
        patch(
            "tldr_scholar.synthesize_style.CorpusCache",
            return_value=MagicMock(get=MagicMock(return_value=None)),
        ),
        patch(
            "tldr_scholar.synthesize_style.ScraperFactory.get_scraper",
            return_value=MagicMock(),
        ),
        patch(
            "tldr_scholar.synthesize_style.build_corpus",
            new_callable=AsyncMock,
            return_value=mock_corpus_result,
        ),
        patch(
            "tldr_scholar.synthesize_style.build_baselines",
            new_callable=AsyncMock,
            return_value=_BASELINES,
        ),
        patch(
            "tldr_scholar.synthesize_style.correlate_against_baselines",
            new_callable=AsyncMock,
            return_value=[_DELTA],
        ),
        patch(
            "tldr_scholar.synthesize_style.aggregate_topic",
            new_callable=AsyncMock,
            return_value=(_TOPIC_PROFILE, True),
        ),
        patch(
            "tldr_scholar.synthesize_style.aggregate_global",
            new_callable=AsyncMock,
            return_value=(_GLOBAL_FIELDS, True),
        ),
        patch(
            "tldr_scholar.synthesize_style.DEFAULT_PERSONA_DIR",
            tmp_path,
        ),
    ):
        await run_synthesis(args)

    output_file = tmp_path / "smoke-persona.yaml"
    assert output_file.exists(), f"Expected persona YAML at {output_file}"
