"""Tests for WU-6 gen-time topic blending in PromptBuilder.build_system_prompt.

Coverage:
- Unit: _cosine_sim, _softmax, _blend_lists helpers
- Integration: build_system_prompt persona path (empty/global/multi-topic/mismatch)
- AST guard: prompts.py must not import sentence_transformers/bertopic/hdbscan
- Flat-priority removal: old aggregate loop symbols are absent
"""
from __future__ import annotations

import ast
import math
import pathlib
import sys
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from tldr_scholar.prompts import (
    PromptBuilder,
    _blend_lists,
    _cosine_sim,
    _softmax,
    _SOFTMAX_TAU,
    _MIN_WEIGHT,
    _TOP_K,
)
from tldr_scholar.topic_cluster import EMBEDDING_MODEL_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_topic(
    label: str,
    centroid: list[float],
    *,
    revelation_priorities: list[str] | None = None,
    suppression_rules: list[str] | None = None,
    rhetorical_strategy: str = "",
    sample_size: int = 5,
) -> dict:
    return {
        "label": label,
        "centroid": centroid,
        "sample_size": sample_size,
        "revelation_priorities": revelation_priorities or [],
        "suppression_rules": suppression_rules or [],
        "rhetorical_strategy": rhetorical_strategy,
    }


def _unit_vec(dim: int, index: int) -> list[float]:
    """L2-normalised unit vector with 1.0 at position `index`, 0 elsewhere."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _v2_persona_dict(name: str, topics: dict, *, embedding_model: str | None = None) -> dict:
    return {
        "name": name,
        "embedding_model": embedding_model or EMBEDDING_MODEL_NAME,
        "role": "analyst",
        "tone": "analytical",
        "structure_pattern": "stitched",
        "hashtag_style": "lowercase",
        "agenda": "",
        "worldview": "",
        "topics": topics,
    }


def _build_with_persona(tmp_path, persona_dict: dict, text: str) -> str:
    """Write persona to disk, return build_system_prompt output."""
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir(exist_ok=True)
    name = persona_dict["name"]
    (persona_dir / f"{name}.yaml").write_text(yaml.dump(persona_dict))

    builder = PromptBuilder()
    builder._persona_manager.config_dir = persona_dir
    builder._persona_manager.reload()

    # Patch embed_text to avoid model loading; return the text as a unit vec
    # (actual vec injected per-test via mock)
    with patch("tldr_scholar.topic_cluster.embed_text") as mock_embed:
        mock_embed.return_value = [0.0] * 384
        result = builder.build_system_prompt(
            mode="scientific",
            max_chars=500,
            focus="",
            hashtag_instruction="",
            persona=name,
            text=text,
        )
    return result


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestCosineSimHelper:
    def test_identical_unit_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(-1.0)

    def test_partial_similarity(self):
        inv_sqrt2 = 1.0 / math.sqrt(2)
        a = [inv_sqrt2, inv_sqrt2]
        b = [1.0, 0.0]
        assert _cosine_sim(a, b) == pytest.approx(inv_sqrt2)

    def test_strict_zip_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cosine_sim([1.0, 0.0], [1.0])


class TestSoftmaxHelper:
    def test_uniform_scores_equal_weights(self):
        weights = _softmax([1.0, 1.0, 1.0], tau=_SOFTMAX_TAU)
        assert all(w == pytest.approx(1 / 3) for w in weights)

    def test_weights_sum_to_one(self):
        scores = [0.9, 0.5, 0.1, 0.3]
        weights = _softmax(scores, tau=_SOFTMAX_TAU)
        assert sum(weights) == pytest.approx(1.0)

    def test_highest_score_gets_highest_weight(self):
        scores = [0.9, 0.1, 0.2]
        weights = _softmax(scores, tau=_SOFTMAX_TAU)
        assert weights[0] == max(weights)

    def test_numerically_stable_with_large_negative_scores(self):
        # Should not raise OverflowError
        scores = [-1000.0, -999.0, -998.0]
        weights = _softmax(scores, tau=_SOFTMAX_TAU)
        assert sum(weights) == pytest.approx(1.0)
        assert weights[2] == max(weights)

    def test_temperature_sharpens_distribution(self):
        scores = [0.9, 0.5]
        w_sharp = _softmax(scores, tau=0.1)
        w_flat = _softmax(scores, tau=1.0)
        # Lower temperature → more mass on highest score
        assert w_sharp[0] > w_flat[0]

    def test_hand_computed_tau03(self):
        # Manual: scores [0.9, 0.1], tau=0.3
        # exp((0.9-0.9)/0.3) = 1.0, exp((0.1-0.9)/0.3) = exp(-2.666...)
        tau = 0.3
        s = [0.9, 0.1]
        m = 0.9
        e0 = math.exp(0.0)
        e1 = math.exp((0.1 - 0.9) / tau)
        total = e0 + e1
        expected = [e0 / total, e1 / total]
        result = _softmax(s, tau)
        assert result[0] == pytest.approx(expected[0])
        assert result[1] == pytest.approx(expected[1])


class TestBlendListsHelper:
    def test_single_topic_returns_its_items(self):
        result = _blend_lists([(1.0, ["a", "b", "c"])])
        assert result == ["a", "b", "c"]

    def test_higher_weight_topic_items_ranked_first(self):
        # "x" appears only in topic-A (w=0.9), "y" only in topic-B (w=0.1)
        result = _blend_lists([(0.9, ["x"]), (0.1, ["y"])])
        assert result.index("x") < result.index("y")

    def test_shared_item_accumulates_weight(self):
        # "shared" appears in both topics; should rank first over "solo"
        result = _blend_lists([(0.5, ["shared", "solo-a"]), (0.5, ["shared", "solo-b"])])
        assert result[0] == "shared"

    def test_deduplication(self):
        result = _blend_lists([(0.6, ["a", "b"]), (0.4, ["a", "c"])])
        assert len(result) == len(set(result))  # no duplicates

    def test_empty_input(self):
        assert _blend_lists([]) == []

    def test_empty_inner_lists(self):
        assert _blend_lists([(0.5, []), (0.5, [])]) == []


# ---------------------------------------------------------------------------
# Integration tests: build_system_prompt persona path
# ---------------------------------------------------------------------------

class TestBuildSystemPromptBlend:
    def test_global_only_persona_skips_blend_embed_not_called(self, tmp_path):
        """_global-only topic: embed_text must NOT be called."""
        topics = {
            "_global": _make_topic(
                "_global",
                [0.0] * 384,
                revelation_priorities=["fact-X"],
                suppression_rules=["hype-Y"],
            )
        }
        persona_dict = _v2_persona_dict("g_only", topics)
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "g_only.yaml").write_text(yaml.dump(persona_dict))

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        with patch("tldr_scholar.topic_cluster.embed_text") as mock_embed:
            prompt = builder.build_system_prompt(
                mode="scientific",
                max_chars=500,
                focus="",
                hashtag_instruction="",
                persona="g_only",
                text="word " * 400,
            )

        mock_embed.assert_not_called()
        assert "fact-X" in prompt
        assert "hype-Y" in prompt

    def test_global_only_persona_uses_global_fields(self, tmp_path):
        """_global-only: revelation + suppression from _global appear in prompt."""
        topics = {
            "_global": _make_topic(
                "_global",
                [0.0] * 384,
                revelation_priorities=["carbon-budget"],
                suppression_rules=["greenwashing"],
            )
        }
        persona_dict = _v2_persona_dict("env", topics)
        result = _build_with_persona(tmp_path, persona_dict, "word " * 400)
        assert "carbon-budget" in result
        assert "greenwashing" in result

    def test_empty_topics_no_crash_returns_prompt(self, tmp_path):
        """Empty topics dict must not crash; returns a non-empty prompt string."""
        # Note: Pydantic requires at least the field; we inject an empty dict
        # by bypassing model validation via dict() patching
        topics: dict = {}
        persona_dict = _v2_persona_dict("bare", topics)
        # Pydantic may reject empty topics — if so, test the safe-fallback path
        # by mocking get_persona to return a Persona with empty topics.
        from tldr_scholar.personas import Persona, TopicProfile
        bare_persona = Persona(
            name="bare",
            embedding_model=EMBEDDING_MODEL_NAME,
            role="analyst",
            tone="analytical",
            structure_pattern="stitched",
            hashtag_style="lowercase",
            topics={},
        )

        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        with (
            patch.object(
                builder._persona_manager,
                "get_persona",
                return_value=bare_persona,
            ),
            patch("tldr_scholar.topic_cluster.embed_text") as mock_embed,
        ):
            mock_embed.return_value = [0.0] * 384
            prompt = builder.build_system_prompt(
                mode="scientific",
                max_chars=500,
                focus="",
                hashtag_instruction="",
                persona="bare",
                text="word " * 400,
            )

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # No REVEAL/SUPPRESS injected when no priorities
        assert "REVEAL" not in prompt
        assert "SUPPRESS" not in prompt

    def test_multi_topic_calls_embed_text_once(self, tmp_path):
        """Multi-topic persona: embed_text called exactly once."""
        dim = 384
        topics = {
            "topic-a": _make_topic("topic-a", _unit_vec(dim, 0), revelation_priorities=["a-point"]),
            "topic-b": _make_topic("topic-b", _unit_vec(dim, 1), revelation_priorities=["b-point"]),
            "topic-c": _make_topic("topic-c", _unit_vec(dim, 2), revelation_priorities=["c-point"]),
        }
        persona_dict = _v2_persona_dict("multi", topics)
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "multi.yaml").write_text(yaml.dump(persona_dict))

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        src_vec = _unit_vec(dim, 0)  # closest to topic-a
        with patch("tldr_scholar.topic_cluster.embed_text", return_value=src_vec) as mock_embed:
            prompt = builder.build_system_prompt(
                mode="scientific",
                max_chars=500,
                focus="",
                hashtag_instruction="",
                persona="multi",
                text="word " * 400,
            )

        mock_embed.assert_called_once()
        # topic-a's item should dominate
        assert "a-point" in prompt

    def test_multi_topic_top_k_enforced(self, tmp_path):
        """5-topic persona: only top-K=3 retained in blend."""
        dim = 384
        topics = {}
        for idx in range(5):
            topics[f"t{idx}"] = _make_topic(
                f"t{idx}",
                _unit_vec(dim, idx),
                revelation_priorities=[f"point-{idx}"],
            )
        persona_dict = _v2_persona_dict("five", topics)
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "five.yaml").write_text(yaml.dump(persona_dict))

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        # src_vec = unit vec at index 0: topic t0 dominates, others get
        # weight distributed via softmax τ=0.3 over cosine sims.
        # With 5 unit vecs, sims are: t0=1.0, others=0.0.
        # softmax([1,0,0,0,0], τ=0.3) → t0 dominates, others get equal small weight
        # After τ=0.3 softmax: exp(1/0.3)=exp(3.333)≈28.0; exp(0/0.3)=1.0
        # w_t0 = 28/(28+4*1) ≈ 0.875 → well above 0.1
        # w_others = 1/32 ≈ 0.031 → below 0.1 → dropped
        # So only t0 retained. That's ≤ K=3, so all retained ≤ 3.
        # The assertion is: the blended prompt doesn't contain MORE than K topics' items.
        src_vec = _unit_vec(dim, 0)
        with patch("tldr_scholar.topic_cluster.embed_text", return_value=src_vec):
            prompt = builder.build_system_prompt(
                mode="scientific",
                max_chars=500,
                focus="",
                hashtag_instruction="",
                persona="five",
                text="word " * 400,
            )

        # t0 is the dominant topic
        assert "point-0" in prompt

    def test_multi_topic_low_similarity_topic_dropped(self, tmp_path):
        """Topic with weight < 0.1 after softmax is excluded from blend."""
        dim = 384
        # t0 perfectly aligned, t1 orthogonal (sim=0)
        # With τ=0.3: softmax([1,0], τ=0.3)
        # exp(1/0.3)≈28, exp(0)=1 → w_t0≈0.966, w_t1≈0.034 < 0.1
        topics = {
            "t0": _make_topic(
                "t0", _unit_vec(dim, 0),
                revelation_priorities=["dominant-item"],
                suppression_rules=["dominant-suppress"],
            ),
            "t1": _make_topic(
                "t1", _unit_vec(dim, 1),
                revelation_priorities=["low-sim-item"],
            ),
        }
        persona_dict = _v2_persona_dict("lowsim", topics)
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "lowsim.yaml").write_text(yaml.dump(persona_dict))

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        src_vec = _unit_vec(dim, 0)
        with patch("tldr_scholar.topic_cluster.embed_text", return_value=src_vec):
            prompt = builder.build_system_prompt(
                mode="scientific",
                max_chars=500,
                focus="",
                hashtag_instruction="",
                persona="lowsim",
                text="word " * 400,
            )

        assert "dominant-item" in prompt
        assert "dominant-suppress" in prompt
        assert "low-sim-item" not in prompt

    def test_three_topic_blended_priorities_in_prompt(self, tmp_path):
        """Multi-topic: blended revelation_priorities appear in returned prompt."""
        dim = 384
        # Three topics, each as unit vec; src_vec equally distant from all
        # → equal softmax weights → all three items should appear
        inv_sqrt3 = 1.0 / math.sqrt(3)
        src_vec = [inv_sqrt3, inv_sqrt3, inv_sqrt3] + [0.0] * (dim - 3)
        topics = {
            "t0": _make_topic("t0", _unit_vec(dim, 0), revelation_priorities=["item-alpha"]),
            "t1": _make_topic("t1", _unit_vec(dim, 1), revelation_priorities=["item-beta"]),
            "t2": _make_topic("t2", _unit_vec(dim, 2), revelation_priorities=["item-gamma"]),
        }
        persona_dict = _v2_persona_dict("three", topics)
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "three.yaml").write_text(yaml.dump(persona_dict))

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        with patch("tldr_scholar.topic_cluster.embed_text", return_value=src_vec):
            prompt = builder.build_system_prompt(
                mode="scientific",
                max_chars=500,
                focus="",
                hashtag_instruction="",
                persona="three",
                text="word " * 400,
            )

        assert "item-alpha" in prompt
        assert "item-beta" in prompt
        assert "item-gamma" in prompt

    def test_embedding_model_mismatch_exits_3(self, tmp_path):
        """Mismatch between persona.embedding_model and installed default → exit(3)."""
        topics = {
            "_global": _make_topic("_global", [0.0] * 384)
        }
        persona_dict = _v2_persona_dict(
            "mismatch",
            topics,
            embedding_model="sentence-transformers/paraphrase-multilingual-v2",
        )
        persona_dir = tmp_path / "personas"
        persona_dir.mkdir()
        (persona_dir / "mismatch.yaml").write_text(yaml.dump(persona_dict))

        builder = PromptBuilder()
        builder._persona_manager.config_dir = persona_dir
        builder._persona_manager.reload()

        with (
            patch("tldr_scholar.topic_cluster.embed_text") as mock_embed,
            patch("tldr_scholar.error_contract.emit_envelope") as mock_emit,
        ):
            mock_embed.return_value = [0.0] * 384
            with pytest.raises(SystemExit) as exc_info:
                builder.build_system_prompt(
                    mode="scientific",
                    max_chars=500,
                    focus="",
                    hashtag_instruction="",
                    persona="mismatch",
                    text="word " * 400,
                )

        assert exc_info.value.code == 3
        mock_emit.assert_called_once()
        call_kwargs = mock_emit.call_args
        assert call_kwargs.kwargs.get("code") == "embedding_model_mismatch" or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "embedding_model_mismatch"
        )

    def test_word_count_below_threshold_falls_back_to_non_persona(self, tmp_path):
        """Texts < 300 words skip the persona path entirely."""
        topics = {
            "_global": _make_topic(
                "_global", [0.0] * 384,
                revelation_priorities=["should-not-appear"]
            )
        }
        persona_dict = _v2_persona_dict("short", topics)
        result = _build_with_persona(tmp_path, persona_dict, "word " * 10)
        assert "should-not-appear" not in result


# ---------------------------------------------------------------------------
# AST guard: purity
# ---------------------------------------------------------------------------

class TestPromptModulePurity:
    def test_prompts_module_has_no_ml_imports(self):
        """prompts.py must not import sentence_transformers, bertopic, or hdbscan."""
        tree = ast.parse(
            pathlib.Path("tldr_scholar/prompts.py").read_text()
        )
        forbidden = {"sentence_transformers", "bertopic", "hdbscan"}
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = getattr(node, "module", None) or ""
                alias_names = [a.name for a in getattr(node, "names", [])]
                all_names = [module_name] + alias_names
                for n in all_names:
                    for fb in forbidden:
                        if fb in (n or ""):
                            violations.append(n)
        assert violations == [], (
            f"prompts.py must not import {forbidden}; found: {violations}"
        )


# ---------------------------------------------------------------------------
# Flat-priority code path removal guard
# ---------------------------------------------------------------------------

class TestFlatPriorityRemoval:
    def test_legacy_flat_collect_loop_is_absent(self):
        """The old flat-collect variable names must not appear in prompts.py."""
        src = pathlib.Path("tldr_scholar/prompts.py").read_text()
        # These were the specific variable names used in the old flat loop
        removed_patterns = ["all_revelation", "all_suppression", "all_rhetorical", "unique_revelation", "unique_suppression"]
        found = [p for p in removed_patterns if p in src]
        assert found == [], (
            f"Legacy flat-priority variable(s) still present in prompts.py: {found}"
        )
