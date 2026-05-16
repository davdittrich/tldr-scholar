"""Tests for deep merge logic in refine_persona."""
from __future__ import annotations

from tldr_scholar.refine_persona import deep_merge

def test_deep_merge_basic():
    base = {"a": 1, "b": {"c": 2}}
    update = {"b": {"d": 3}, "e": 4}
    result = deep_merge(base, update)
    assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

def test_deep_merge_overwrite():
    base = {"a": 1, "b": 2}
    update = {"a": 3}
    result = deep_merge(base, update)
    assert result == {"a": 3, "b": 2}
