"""Pre-flight checks for tldr-scholar pipeline.

Runs once at synthesize_style main entry, before the sampling/scraping stage.
Warns (never errors) about missing model caches.
"""
from __future__ import annotations

import os
from pathlib import Path
from loguru import logger

from tldr_scholar.topic_cluster import EMBEDDING_MODEL_NAME

# ---------------------------------------------------------------------------
# Cache detection
# ---------------------------------------------------------------------------

def _model_is_cached() -> bool:
    """Return True if the sentence-transformers model is already on disk.

    Checks the HuggingFace hub cache (default ~/.cache/huggingface/hub/).
    Falls back to sentence-transformers legacy dir (~/.cache/sentence_transformers/).
    """
    # HuggingFace Hub cache: model slug → snapshots/
    hf_cache = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"

    # slug: "sentence-transformers/all-MiniLM-L6-v2"
    # HF stores as "models--sentence-transformers--all-MiniLM-L6-v2"
    model_slug = EMBEDDING_MODEL_NAME.replace("/", "--")
    hf_dir = hf_cache / f"models--{model_slug}"
    if hf_dir.exists():
        return True

    # Legacy sentence-transformers cache
    st_cache = Path.home() / ".cache" / "sentence_transformers"
    st_dir = st_cache / EMBEDDING_MODEL_NAME
    if st_dir.exists():
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_embedding_model_cached() -> None:
    """Warn if the sentence-transformers embedding model is not cached on disk.

    This is a soft check only — no exception is raised, no exit is triggered.
    The actual model download (if needed) happens lazily inside topic_cluster._get_model().
    """
    if not _model_is_cached():
        logger.warning(
            f"Embedding model '{EMBEDDING_MODEL_NAME}' not found in local cache. "
            "First run will trigger an automatic download (~50 MB). "
            "Set HF_HOME to control cache location."
        )
