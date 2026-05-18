"""Tests for 3-baseline correlator (WU-3, tldr-scholar-nwl.3)."""
from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock

import pytest

from tldr_scholar.personas import DeltaRecord
from tldr_scholar.source_baseline import SourceBaselines
from tldr_scholar.correlator import correlate_against_baselines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POST_TEXT = "Compound X seems effective; I skimmed their data."

_BASELINES_CLAIMS_ONLY = SourceBaselines(
    claims=["Compound X reduces inflammation by 40%.", "p=0.003", "N=120"],
    extractive_summary=None,
    abstractive_summary=None,
)

_BASELINES_ALL_THREE = SourceBaselines(
    claims=["Compound X reduces inflammation by 40%.", "p=0.003"],
    extractive_summary=(
        "Researchers found compound X reduces inflammation. "
        "The p-value was 0.003. Sample size was 120."
    ),
    abstractive_summary=(
        "Compound X reduces inflammation by 40% (p=0.003, N=120). "
        "IL-6 pathway is implicated. Further studies needed."
    ),
)

_BASELINES_ALL_NONE = SourceBaselines(
    claims=None,
    extractive_summary=None,
    abstractive_summary=None,
)


def _make_correlation_response(baseline_type: str) -> str:
    return textwrap.dedent(f"""\
        - statement_id: s1
          status: shared
          intent: Post acknowledges the finding.
        - statement_id: s2
          status: distorted
          intent: Slight framing shift.
    """).strip()


# ---------------------------------------------------------------------------
# Test: DeltaRecord shape & Literal types
# ---------------------------------------------------------------------------

class TestDeltaRecordShape:
    def test_literal_status_values(self):
        """DeltaRecord status values must be Literal strings, not Enum."""
        rec = DeltaRecord(
            baseline_type="claims",
            statements=["stmt1", "stmt2"],
            status_per_statement=["shared", "suppressed"],
            intent="test",
        )
        assert rec.status_per_statement[0] == "shared"
        assert isinstance(rec.status_per_statement[0], str)

    def test_baseline_type_literal(self):
        for bt in ("claims", "extractive", "abstractive"):
            rec = DeltaRecord(
                baseline_type=bt,
                statements=["x"],
                status_per_statement=["shared"],
            )
            assert rec.baseline_type == bt

    def test_intent_optional(self):
        rec = DeltaRecord(
            baseline_type="extractive",
            statements=["x"],
            status_per_statement=["shared"],
        )
        assert rec.intent is None


# ---------------------------------------------------------------------------
# Test: correlate_against_baselines — claims-only baseline
# ---------------------------------------------------------------------------

class TestCorrelateClaimsOnly:
    @pytest.mark.asyncio
    async def test_claims_delta_record_produced(self):
        """Claims baseline → one DeltaRecord with baseline_type='claims'."""
        llm_call = AsyncMock(return_value=_make_correlation_response("claims"))
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_CLAIMS_ONLY, llm_call=llm_call
        )

        assert len(records) == 1
        assert records[0].baseline_type == "claims"

    @pytest.mark.asyncio
    async def test_post_text_wrapped_in_untrusted_content(self):
        """CORRELATION_PROMPT call must wrap post_text in <untrusted_content>."""
        calls = []

        async def capturing_llm(prompt: str) -> str:
            calls.append(prompt)
            return _make_correlation_response("claims")

        await correlate_against_baselines(
            _POST_TEXT, _BASELINES_CLAIMS_ONLY, llm_call=capturing_llm
        )

        assert calls, "LLM must be called"
        assert "<untrusted_content>" in calls[0], "post_text must be wrapped in <untrusted_content>"
        assert _POST_TEXT in calls[0], "post_text must appear in prompt"

    @pytest.mark.asyncio
    async def test_delta_record_statements_match_claims(self):
        """statements field of DeltaRecord must list the atomic claims."""
        llm_call = AsyncMock(return_value=_make_correlation_response("claims"))
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_CLAIMS_ONLY, llm_call=llm_call
        )

        assert records[0].statements == _BASELINES_CLAIMS_ONLY.claims

    @pytest.mark.asyncio
    async def test_status_per_statement_valid_literals(self):
        """All status values in DeltaRecord must be valid Literal values."""
        llm_call = AsyncMock(return_value=_make_correlation_response("claims"))
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_CLAIMS_ONLY, llm_call=llm_call
        )

        valid = {"shared", "suppressed", "distorted"}
        for rec in records:
            for s in rec.status_per_statement:
                assert s in valid, f"Invalid status: {s}"


# ---------------------------------------------------------------------------
# Test: all-3 baselines
# ---------------------------------------------------------------------------

class TestCorrelateAllThree:
    @pytest.mark.asyncio
    async def test_three_delta_records_for_all_baselines(self):
        """All 3 baselines populated → 3 DeltaRecords (one per baseline_type)."""
        llm_call = AsyncMock(return_value=_make_correlation_response("any"))
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_ALL_THREE, llm_call=llm_call
        )

        baseline_types = {r.baseline_type for r in records}
        assert "claims" in baseline_types
        assert "extractive" in baseline_types
        assert "abstractive" in baseline_types

    @pytest.mark.asyncio
    async def test_per_baseline_failure_drops_only_that_record(self):
        """If extractive correlation fails, only extractive DeltaRecord is dropped."""
        call_count = 0

        async def selective_fail(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # Let claims=1st call and abstractive=3rd pass; fail extractive=2nd
            if call_count == 2:
                raise RuntimeError("extractive corr fail")
            return _make_correlation_response("any")

        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_ALL_THREE, llm_call=selective_fail
        )

        types = {r.baseline_type for r in records}
        assert "claims" in types
        assert "abstractive" in types
        assert "extractive" not in types

    @pytest.mark.asyncio
    async def test_all_baselines_none_returns_empty_list(self):
        """All None baselines → correlate returns []."""
        llm_call = AsyncMock()
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_ALL_NONE, llm_call=llm_call
        )

        assert records == []
        llm_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_correlations_fail_returns_empty_list(self):
        """All LLM correlation calls fail → returns []."""
        llm_call = AsyncMock(side_effect=RuntimeError("total corr fail"))
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_CLAIMS_ONLY, llm_call=llm_call
        )

        assert records == []


# ---------------------------------------------------------------------------
# Test: sentence splitting for extractive/abstractive statements
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    @pytest.mark.asyncio
    async def test_extractive_statements_split_into_sentences(self):
        """Extractive/abstractive summaries split into individual sentence statements."""
        llm_call = AsyncMock(return_value=_make_correlation_response("any"))
        records = await correlate_against_baselines(
            _POST_TEXT, _BASELINES_ALL_THREE, llm_call=llm_call
        )

        ext_rec = next((r for r in records if r.baseline_type == "extractive"), None)
        assert ext_rec is not None
        # extractive_summary has 3 sentences → statements should have 3 items
        assert len(ext_rec.statements) >= 2, "Should split into multiple sentences"
