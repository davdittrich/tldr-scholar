"""Tests for 3-baseline source decomposition (WU-3, tldr-scholar-nwl.3)."""
from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, patch

import pytest

from tldr_scholar.source_baseline import build_baselines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = textwrap.dedent("""\
    Researchers found that compound X reduces inflammation by 40% in mice.
    The p-value was 0.003. Further studies are needed. Sample size was N=120.
    The mechanism involves IL-6 pathway suppression. Results were reproducible.
""").strip()

_MOCK_CLAIMS_RESPONSE = textwrap.dedent("""\
    - id: claim_1
      claim: Compound X reduces inflammation by 40% in mice.
    - id: claim_2
      claim: The p-value was 0.003.
    - id: claim_3
      claim: Sample size was N=120.
""").strip()

_MOCK_ABSTRACTIVE_RESPONSE = (
    "Compound X reduces inflammation by 40% in mice (p=0.003, N=120). "
    "The mechanism involves IL-6 pathway suppression. "
    "Further studies are needed to confirm reproducibility."
)


# ---------------------------------------------------------------------------
# Test: default mode (full=False) — only claims attempted
# ---------------------------------------------------------------------------

class TestBuildBaselinesDefault:
    @pytest.mark.asyncio
    async def test_claims_attempted_by_default(self):
        """full=False → LLM called with DECOMPOSITION_PROMPT; extractive/abstractive skipped."""
        llm_call = AsyncMock(return_value=_MOCK_CLAIMS_RESPONSE)
        result = await build_baselines(_SAMPLE_TEXT, full=False, llm_call=llm_call)

        assert llm_call.call_count == 1, "Only one LLM call in claims-only mode"
        # claims should be populated (parsed from YAML)
        assert result.claims is not None
        assert len(result.claims) >= 1

    @pytest.mark.asyncio
    async def test_extractive_abstractive_none_by_default(self):
        """full=False → extractive and abstractive remain None."""
        llm_call = AsyncMock(return_value=_MOCK_CLAIMS_RESPONSE)
        result = await build_baselines(_SAMPLE_TEXT, full=False, llm_call=llm_call)

        assert result.extractive_summary is None
        assert result.abstractive_summary is None

    @pytest.mark.asyncio
    async def test_claims_none_on_llm_failure(self):
        """LLM raises → claims=None, no crash."""
        llm_call = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        result = await build_baselines(_SAMPLE_TEXT, full=False, llm_call=llm_call)

        assert result.claims is None

    @pytest.mark.asyncio
    async def test_claims_none_on_bad_yaml(self):
        """LLM returns non-list garbage → claims=None, no crash."""
        llm_call = AsyncMock(return_value="NOT VALID YAML {{{{")
        result = await build_baselines(_SAMPLE_TEXT, full=False, llm_call=llm_call)

        assert result.claims is None


# ---------------------------------------------------------------------------
# Test: full mode (full=True) — all 3 attempted
# ---------------------------------------------------------------------------

class TestBuildBaselinesFullMode:
    @pytest.mark.asyncio
    async def test_all_three_attempted(self):
        """full=True → LLM called twice (claims + abstractive) + sumy run."""
        llm_call = AsyncMock(return_value=_MOCK_CLAIMS_RESPONSE)
        # patch abstractive call separately via side_effect
        responses = [_MOCK_CLAIMS_RESPONSE, _MOCK_ABSTRACTIVE_RESPONSE]
        llm_call.side_effect = responses
        result = await build_baselines(_SAMPLE_TEXT, full=True, llm_call=llm_call)

        assert llm_call.call_count == 2, "Claims LLM + abstractive LLM"
        assert result.claims is not None
        assert result.extractive_summary is not None  # sumy ran (CPU-only)
        assert result.abstractive_summary is not None

    @pytest.mark.asyncio
    async def test_extractive_deterministic(self):
        """Sumy LexRank on a fixed text returns the same result across calls."""
        llm_call = AsyncMock(side_effect=["", ""])  # empty → both LLM paths fail/None
        result1 = await build_baselines(_SAMPLE_TEXT, full=True, llm_call=llm_call)
        llm_call2 = AsyncMock(side_effect=["", ""])
        result2 = await build_baselines(_SAMPLE_TEXT, full=True, llm_call=llm_call2)

        assert result1.extractive_summary == result2.extractive_summary

    @pytest.mark.asyncio
    async def test_abstractive_uses_neutral_summary_prompt(self):
        """Abstractive call must contain NEUTRAL_SUMMARY_PROMPT tokens and <untrusted_content>."""
        from tldr_scholar.prompts import NEUTRAL_SUMMARY_PROMPT

        calls = []

        async def capturing_llm(prompt: str) -> str:
            calls.append(prompt)
            return _MOCK_ABSTRACTIVE_RESPONSE

        result = await build_baselines(_SAMPLE_TEXT, full=True, llm_call=capturing_llm)

        abstractive_calls = [c for c in calls if "<untrusted_content>" in c]
        assert abstractive_calls, "At least one LLM call must wrap content in <untrusted_content>"
        assert result.abstractive_summary is not None
        # Verify the abstractive call uses tokens from NEUTRAL_SUMMARY_PROMPT
        assert any(NEUTRAL_SUMMARY_PROMPT[:30] in c for c in abstractive_calls), (
            "Abstractive LLM call must begin with NEUTRAL_SUMMARY_PROMPT"
        )

    @pytest.mark.asyncio
    async def test_abstractive_failure_isolated(self):
        """Abstractive LLM raises → abstractive=None; claims+extractive still populated."""
        call_count = 0

        async def selective_fail(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MOCK_CLAIMS_RESPONSE  # first call = claims
            raise RuntimeError("abstractive fail")  # second call = abstractive

        result = await build_baselines(_SAMPLE_TEXT, full=True, llm_call=selective_fail)

        assert result.claims is not None
        assert result.extractive_summary is not None
        assert result.abstractive_summary is None

    @pytest.mark.asyncio
    async def test_all_three_fail_returns_all_none(self):
        """All sources fail → SourceBaselines with all None fields."""
        llm_call = AsyncMock(side_effect=RuntimeError("total failure"))
        with patch("tldr_scholar.source_baseline._extractive_summarize", side_effect=RuntimeError("sumy fail")):
            result = await build_baselines(_SAMPLE_TEXT, full=True, llm_call=llm_call)

        assert result.claims is None
        assert result.extractive_summary is None
        assert result.abstractive_summary is None

    @pytest.mark.asyncio
    async def test_decomposition_prompt_wraps_untrusted_content(self):
        """Claims LLM call must wrap source text in <untrusted_content>."""
        calls = []

        async def capturing_llm(prompt: str) -> str:
            calls.append(prompt)
            return _MOCK_CLAIMS_RESPONSE

        await build_baselines(_SAMPLE_TEXT, full=False, llm_call=capturing_llm)

        assert calls, "LLM must be called"
        assert "<untrusted_content>" in calls[0], "Prompt must wrap source in <untrusted_content>"
        assert _SAMPLE_TEXT in calls[0], "Source text must appear in prompt"
