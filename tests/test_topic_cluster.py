"""Tests for topic_cluster lazy-singleton and cluster_posts interface."""
from __future__ import annotations

import ast
import importlib
import sys
import types


def _import_fresh(name: str):
    """Import module, evicting from sys.modules first."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_import_does_not_trigger_model_download(monkeypatch):
    """Importing topic_cluster must NOT call SentenceTransformer() or download anything."""
    download_called = []

    # Intercept sentence_transformers at import time
    fake_st = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, *a, **kw):
            download_called.append(("SentenceTransformer", a, kw))

    fake_st.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    # Clear hdbscan + bertopic too so they don't error
    for mod in list(sys.modules.keys()):
        if mod.startswith(("hdbscan", "bertopic", "tldr_scholar.topic_cluster")):
            sys.modules.pop(mod, None)

    import tldr_scholar.topic_cluster  # noqa: F401

    assert download_called == [], (
        f"import topic_cluster triggered model download: {download_called}"
    )


def test_module_level_model_is_none_at_import(monkeypatch):
    """_MODEL global must be None immediately after import."""
    fake_st = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, *a, **kw):
            pass

    fake_st.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    for mod in list(sys.modules.keys()):
        if mod.startswith(("tldr_scholar.topic_cluster",)):
            sys.modules.pop(mod, None)

    import tldr_scholar.topic_cluster as tc

    assert tc._MODEL is None


def test_topic_cluster_does_not_import_prompts():
    """AST test: topic_cluster must NOT import from tldr_scholar.prompts."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "tldr_scholar" / "topic_cluster.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "prompts" not in module, (
                    f"topic_cluster imports from prompts: {ast.dump(node)}"
                )
            else:
                for alias in node.names:
                    assert "prompts" not in alias.name, (
                        f"topic_cluster imports prompts: {ast.dump(node)}"
                    )


def test_embed_text_returns_list_of_floats(monkeypatch):
    """embed_text returns a list of floats when model is mocked."""
    import numpy as np

    fake_st = types.ModuleType("sentence_transformers")

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            n = len(texts) if isinstance(texts, list) else 1
            return np.zeros((n, 384), dtype=np.float32)

    class FakeST:
        def __init__(self, *a, **kw):
            pass

        def __new__(cls, *a, **kw):
            return FakeModel()

    fake_st.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    for mod in list(sys.modules.keys()):
        if mod.startswith("tldr_scholar.topic_cluster"):
            sys.modules.pop(mod, None)

    import tldr_scholar.topic_cluster as tc

    # Patch _get_model to return fake model without triggering download
    tc._MODEL = FakeModel()
    result = tc.embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_embed_batch_returns_list_of_vectors(monkeypatch):
    """embed_batch(['a','b']) returns a list of two 384-d vectors."""
    import numpy as np

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return np.zeros((len(texts), 384), dtype=np.float32)

    for mod in list(sys.modules.keys()):
        if mod.startswith("tldr_scholar.topic_cluster"):
            sys.modules.pop(mod, None)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = lambda *a, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    import tldr_scholar.topic_cluster as tc

    tc._MODEL = FakeModel()
    result = tc.embed_batch(["hello", "world"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(result[0]) == 384


def test_cluster_posts_returns_labels_and_centroids(monkeypatch):
    """cluster_posts returns (labels: list[str], centroids: dict[str, list[float]])."""
    import numpy as np

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return np.random.default_rng(0).standard_normal((len(texts), 384)).astype(np.float32)

    for mod in list(sys.modules.keys()):
        if mod.startswith(("tldr_scholar.topic_cluster", "hdbscan", "bertopic")):
            sys.modules.pop(mod, None)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = lambda *a, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    import tldr_scholar.topic_cluster as tc

    tc._MODEL = FakeModel()

    posts = [f"post text number {i}" for i in range(20)]
    labels, centroids = tc.cluster_posts(posts, seed=42)

    assert isinstance(labels, list)
    assert len(labels) == 20
    assert all(isinstance(lb, str) for lb in labels)

    assert isinstance(centroids, dict)
    for key, vec in centroids.items():
        assert isinstance(key, str)
        assert isinstance(vec, list)
        assert len(vec) == 384


def test_hdbscan_failure_emits_warn_envelope(monkeypatch, capsys):
    """When HDBSCAN fails, should emit warn envelope before fallback."""
    import json
    import sys
    import types
    
    # Setup fake ST
    fake_st = types.ModuleType("sentence_transformers")
    class FakeModel:
        def encode(self, texts, **kw):
            import numpy as np
            return [np.ones(384) for _ in texts]
    fake_st.SentenceTransformer = lambda *a, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    
    # Mock HDBSCAN to fail
    fake_hdbscan = types.ModuleType("hdbscan")
    class FailingHDBSCAN:
        def __init__(self, **kw):
            pass
        def fit_predict(self, X):
            raise RuntimeError("HDBSCAN clustering failed")
    fake_hdbscan.HDBSCAN = FailingHDBSCAN
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)
    
    # Clear module cache and reimport
    for mod in list(sys.modules.keys()):
        if mod.startswith(("tldr_scholar.topic_cluster", "bertopic")):
            sys.modules.pop(mod, None)
    
    import tldr_scholar.topic_cluster as tc
    tc._MODEL = FakeModel()
    
    posts = ["hello world", "foo bar"]
    labels, centroids = tc.cluster_posts(posts)
    
    # Check stderr for warn envelope
    captured = capsys.readouterr()
    envelope_lines = [l for l in captured.err.split('\n') if l.strip()]
    assert len(envelope_lines) > 0, "Expected warn envelope on stderr"
    
    env = json.loads(envelope_lines[0])
    assert env["level"] == "warn"
    assert env["stage"] == "cluster"
    assert env["code"] == "hdbscan_failed"


def test_bertopic_failure_emits_warn_envelope(monkeypatch, capsys):
    """When bertopic labeling fails, should emit warn envelope before fallback."""
    import json
    import sys
    import types
    
    # Setup fake ST
    fake_st = types.ModuleType("sentence_transformers")
    class FakeModel:
        def encode(self, texts, **kw):
            import numpy as np
            return [np.ones(384) for _ in texts]
    fake_st.SentenceTransformer = lambda *a, **kw: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    
    # Mock HDBSCAN to succeed but return multiple clusters
    fake_hdbscan = types.ModuleType("hdbscan")
    class WorkingHDBSCAN:
        def __init__(self, **kw):
            pass
        def fit_predict(self, X):
            import numpy as np
            return np.array([0, 1, 0, 1, 2])
    fake_hdbscan.HDBSCAN = WorkingHDBSCAN
    monkeypatch.setitem(sys.modules, "hdbscan", fake_hdbscan)
    
    # Clear module cache and reimport
    for mod in list(sys.modules.keys()):
        if mod.startswith(("tldr_scholar.topic_cluster", "bertopic")):
            sys.modules.pop(mod, None)
    
    import tldr_scholar.topic_cluster as tc
    tc._MODEL = FakeModel()
    
    # Patch _label_clusters to fail
    original_label = tc._label_clusters
    def failing_label(*a, **kw):
        raise ValueError("bertopic labeling failed")
    tc._label_clusters = failing_label
    
    posts = ["hello", "world", "foo", "bar", "baz"]
    labels, centroids = tc.cluster_posts(posts)
    
    # Check stderr for warn envelope
    captured = capsys.readouterr()
    envelope_lines = [l for l in captured.err.split('\n') if l.strip()]
    assert any(
        json.loads(l).get("code") == "bertopic_failed"
        for l in envelope_lines
        if l.strip()
    ), f"Expected bertopic_failed envelope. Got: {envelope_lines}"


def test_label_collision_resolved_via_suffix():
    """Test that duplicate cluster labels are disambiguated with numeric suffixes."""
    from collections import Counter

    # Simulate the labeling logic with a collision scenario
    # Two clusters produce identical top-3 terms
    cluster_terms = {
        0: ["econ", "labor", "wage"],
        1: ["econ", "labor", "wage"],  # Collision!
        2: ["trade", "deal", "pact"],
    }

    # Apply the fix: track seen labels and append suffix
    seen = Counter()
    result = {}
    for cid, top_terms in cluster_terms.items():
        label = "+".join(top_terms) if top_terms else f"topic_{cid}"
        seen[label] += 1
        if seen[label] > 1:
            label = f"{label}_{seen[label]}"
        result[cid] = label

    # Verify labels are unique
    labels_list = list(result.values())
    assert len(labels_list) == len(set(labels_list)), (
        f"Label collision detected: {labels_list}"
    )

    # Verify the expected labels
    assert result[0] == "econ+labor+wage"
    assert result[1] == "econ+labor+wage_2"
    assert result[2] == "trade+deal+pact"

