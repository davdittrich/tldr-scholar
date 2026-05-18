"""Topic clustering for tldr-scholar persona pipeline.

Provides a lazy-loaded sentence-transformers singleton for embedding posts,
HDBSCAN clustering, and bertopic c-TF-IDF auto-labeling.

IMPORTANT: Importing this module MUST NOT trigger any model download.
The model is initialized on first call to _get_model() or any embed_*/cluster_posts call.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_MODEL = None  # initialized on first use; import MUST NOT touch this


def _get_model():
    """Return (and lazy-initialize) the sentence-transformers singleton."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


# ---------------------------------------------------------------------------
# Free embedding functions
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns a 384-d normalized vector as list[float]."""
    import numpy as np
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)
    return vec[0].tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list of 384-d normalized vectors."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_posts(
    posts: list[str],
    seed: int | None = None,
) -> tuple[list[str], dict[str, list[float]]]:
    """Cluster posts into topic groups.

    Args:
        posts: List of post text strings.
        seed:  Random seed for HDBSCAN reproducibility.

    Returns:
        (labels, centroids) where:
        - labels: list[str] of length len(posts). Each entry is a topic key
          (e.g. "economics+labor+wage") or "_unclustered" for noise.
        - centroids: dict[topic_key -> 384-d normalized float list]
    """
    import numpy as np

    if not posts:
        return [], {}

    embeddings = embed_batch(posts)
    emb_arr = np.array(embeddings, dtype=np.float32)

    # --- HDBSCAN clustering ---
    try:
        import hdbscan  # type: ignore
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric="euclidean",  # cosine not available for all hdbscan builds; use euclidean on normalized vecs
            core_dist_n_jobs=1,
        )
        # NOTE: hdbscan.HDBSCAN does not accept a random_state / seed parameter in all
        # released versions (the tree approximation step is deterministic for euclidean
        # metric anyway).  We intentionally do NOT mutate np.random.seed() here —
        # callers that need reproducible downstream operations should manage their own RNG.
        raw_labels: list[int] = clusterer.fit_predict(emb_arr).tolist()
    except Exception:
        # Fallback: single global cluster
        raw_labels = [0] * len(posts)

    # Unique cluster ids (exclude noise = -1)
    unique_ids = sorted({lb for lb in raw_labels if lb >= 0})

    if not unique_ids:
        # HDBSCAN found 0 real clusters → single _global fallback
        global_centroid = _normalize(emb_arr.mean(axis=0)).tolist()
        labels = ["_global"] * len(posts)
        return labels, {"_global": global_centroid}

    # --- Auto-label each cluster with bertopic c-TF-IDF top-3 terms ---
    cluster_labels_map: dict[int, str] = {}
    try:
        cluster_labels_map = _label_clusters(posts, raw_labels, unique_ids)
    except Exception:
        # Fallback: numeric labels
        for cid in unique_ids:
            cluster_labels_map[cid] = f"topic_{cid}"

    # Map each post to its string label
    labels: list[str] = []
    for raw_lb in raw_labels:
        if raw_lb == -1:
            labels.append("_unclustered")
        else:
            labels.append(cluster_labels_map.get(raw_lb, f"topic_{raw_lb}"))

    # --- Build centroids ---
    centroids: dict[str, list[float]] = {}
    all_labels_set = set(labels)
    for topic_key in all_labels_set:
        idxs = [i for i, lb in enumerate(labels) if lb == topic_key]
        cluster_vecs = emb_arr[idxs]
        centroid = _normalize(cluster_vecs.mean(axis=0)).tolist()
        centroids[topic_key] = centroid

    return labels, centroids


def _normalize(vec) -> "np.ndarray":  # type: ignore
    """L2-normalize a 1-D numpy array."""
    import numpy as np
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def _label_clusters(
    posts: list[str],
    raw_labels: list[int],
    unique_ids: list[int],
) -> dict[int, str]:
    """Use bertopic c-TF-IDF to label each cluster with top-3 terms."""
    from bertopic.vectorizers import ClassTfidfTransformer  # type: ignore
    from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

    # Build document-per-cluster representation
    docs_per_cluster: dict[int, list[str]] = {cid: [] for cid in unique_ids}
    for i, lb in enumerate(raw_labels):
        if lb >= 0:
            docs_per_cluster[lb].append(posts[i])

    # One concatenated doc per cluster for TF-IDF
    cluster_docs = [" ".join(docs_per_cluster[cid]) for cid in unique_ids]

    vectorizer = CountVectorizer(stop_words="english", max_features=10_000)
    X = vectorizer.fit_transform(cluster_docs)

    ctfidf = ClassTfidfTransformer()
    ctfidf_matrix = ctfidf.fit_transform(X)

    vocab = vectorizer.get_feature_names_out()
    result: dict[int, str] = {}
    for idx, cid in enumerate(unique_ids):
        row = ctfidf_matrix[idx].toarray().flatten()
        top_idxs = row.argsort()[::-1][:3]
        top_terms = [vocab[j] for j in top_idxs if row[j] > 0]
        label = "+".join(top_terms) if top_terms else f"topic_{cid}"
        result[cid] = label

    return result
