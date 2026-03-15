"""
VidEngram Memory Visualizer
After ingest completes, embed all memories, reduce to 2D with t-SNE,
and save a scatter plot to a fixed path.

Embedding strategy (with automatic fallback):
  1. Qwen3-Embedding-4B via vLLM at localhost:8003  (preferred)
  2. TF-IDF + Truncated SVD                          (fallback when server unavailable)

NOTE: vLLM 0.15.0 has a known bug with Qwen3-Embedding where ANY call to
/v1/embeddings crashes the engine (msgspec.ValidationError on tok_pooling_type='ALL').
Until vLLM is upgraded, the code automatically falls back to TF-IDF.
"""
import logging
from pathlib import Path

import numpy as np
import requests

from .utils import ConsolidatedMemory

logger = logging.getLogger("videngram.visualizer")

# Fixed output path — overwritten on every run
OUTPUT_PATH = Path("./output/memory_tsne.png")

# Color palette per memory type
TYPE_COLORS = {
    "segment":         "#4C9BE8",  # blue
    "episode_summary": "#E87B4C",  # orange
    "entity_register": "#5DBE6E",  # green
    "speech":          "#B07FD4",  # purple
}
TYPE_LABELS = {
    "segment":         "Segment",
    "episode_summary": "Episode Summary",
    "entity_register": "Entity Register",
    "speech":          "Speech",
}


def _try_embed_remote(
    texts: list[str],
    base_url: str,
    model: str,
) -> np.ndarray | None:
    """Attempt to embed texts via the remote vLLM server, one at a time.

    Returns an (N, D) float32 array on success, or None on any failure.

    NOTE: vLLM 0.15.0 crashes the engine process on ANY /v1/embeddings call
    when using Qwen3-Embedding due to a msgspec serialization bug with
    tok_pooling_type='ALL'. This function catches the resulting 500 error
    and returns None so the caller can fall back to TF-IDF.
    """
    url = f"{base_url.rstrip('/')}/embeddings"
    vectors: list[list[float]] = []

    for i, text in enumerate(texts):
        try:
            resp = requests.post(
                url,
                json={"model": model, "input": text},
                timeout=60,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["data"][0]["embedding"])
        except Exception as e:
            logger.warning(
                f"Remote embedding failed on item {i} (vLLM 0.15.0 bug with "
                f"tok_pooling_type='ALL' — upgrade vLLM to fix): {e}"
            )
            return None  # abort immediately; server may have crashed

        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i + 1}/{len(texts)} memories...")

    return np.array(vectors, dtype=np.float32)


def _embed_with_tfidf(texts: list[str]) -> np.ndarray:
    """TF-IDF + Truncated SVD vectorization — no remote server required.

    Produces dense vectors suitable for t-SNE. Dimensionality is capped at
    min(50, n_texts-1) to keep t-SNE well-behaved even with few memories.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(max_features=1000, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)

    n_components = min(50, tfidf.shape[1], max(1, len(texts) - 1))
    if n_components > 1 and tfidf.shape[0] > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        vectors = svd.fit_transform(tfidf)
    else:
        vectors = tfidf.toarray()

    return vectors.astype(np.float32)


def generate_tsne_plot(
    memories: list[ConsolidatedMemory],
    embedding_base_url: str = "http://localhost:8003/v1",
    embedding_model: str = "Qwen/Qwen3-Embedding-4B",
) -> Path | None:
    """Embed memories, run t-SNE, and save the scatter plot.

    Tries Qwen3-Embedding-4B first; falls back to TF-IDF if unavailable.

    Args:
        memories: All ConsolidatedMemory objects from the current ingest run.
        embedding_base_url: Base URL of the OpenAI-compatible embeddings server.
        embedding_model: Model name to pass in the request.

    Returns:
        Path to the saved PNG, or None if skipped/failed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as e:
        logger.warning(f"Visualization skipped — missing dependency: {e}")
        return None

    if not memories:
        logger.warning("Visualization skipped — no memories to plot.")
        return None

    n = len(memories)
    if n < 2:
        logger.warning("Visualization skipped — need at least 2 memories for t-SNE.")
        return None

    texts = [m.content for m in memories]
    types = [m.memory_type for m in memories]

    # --- Embeddings: try remote, fall back to TF-IDF ---
    logger.info(
        f"Embedding {n} memories via {embedding_base_url} ({embedding_model})..."
    )
    embeddings = _try_embed_remote(texts, embedding_base_url, embedding_model)

    if embeddings is None:
        logger.info("Falling back to TF-IDF vectorization for t-SNE.")
        embeddings = _embed_with_tfidf(texts)
        embed_label = "TF-IDF (Qwen3-Embedding unavailable — vLLM 0.15.0 bug)"
    else:
        embed_label = embedding_model

    # --- t-SNE ---
    perplexity = min(30, max(1, n - 1))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca" if n >= 4 else "random",
    )
    try:
        coords = TSNE(**tsne_kwargs, max_iter=1000).fit_transform(embeddings)
    except TypeError:
        coords = TSNE(**tsne_kwargs, n_iter=1000).fit_transform(embeddings)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")

    present_types = [t for t in TYPE_COLORS if t in set(types)]

    for mtype in present_types:
        mask = [i for i, t in enumerate(types) if t == mtype]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(
            xs, ys,
            c=TYPE_COLORS[mtype],
            label=TYPE_LABELS.get(mtype, mtype),
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title("Memory Embedding Visualization (t-SNE)", fontsize=14, pad=12)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    fig.text(
        0.5, 0.01, f"Embedding: {embed_label}",
        ha="center", fontsize=7, color="#888888",
    )
    ax.legend(
        title="Memory Type",
        title_fontsize=9,
        fontsize=9,
        framealpha=0.9,
        loc="best",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # --- Save (always overwrite) ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"t-SNE plot saved to {OUTPUT_PATH}  "
        f"(n={n}, perplexity={perplexity}, embed={embed_label})"
    )
    return OUTPUT_PATH
