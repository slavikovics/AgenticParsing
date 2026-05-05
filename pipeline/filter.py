"""
filter.py — cosine similarity scoring, threshold filtering, and top-K cap.
"""

import logging

import numpy as np

from .models import Chunk

log = logging.getLogger("pipeline")


def score_chunks(
    chunks: list[Chunk],
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> list[Chunk]:
    """
    Score each chunk against all relevance queries, keep the max score.

    Both embedding matrices must be L2-normalised (cosine sim = dot product).
    Shape: doc_embeddings (n_chunks, dim), query_embeddings (n_queries, dim)
    Result: each chunk.similarity = max cosine similarity across all queries.
    """
    scores = doc_embeddings @ query_embeddings.T  # (n_chunks, n_queries)
    max_scores = scores.max(axis=1)

    for chunk, score in zip(chunks, max_scores):
        chunk.similarity = float(score)

    return chunks


def filter_chunks(
    chunks: list[Chunk],
    min_score: float = 0.0,
    top_k: int = 999_999,
) -> tuple[list[Chunk], dict]:
    """
    Filter and rank chunks by similarity score.

    Applies min_score threshold first, then caps at top_k.
    Returns filtered chunks and a stats dict for logging.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.similarity, reverse=True)

    above_threshold = [c for c in sorted_chunks if c.similarity >= min_score]
    final = above_threshold[:top_k]

    stats = {
        "total": len(chunks),
        "above_threshold": len(above_threshold),
        "kept": len(final),
        "dropped": len(chunks) - len(final),
        "top_score": final[0].similarity if final else 0.0,
        "bot_score": final[-1].similarity if final else 0.0,
        "min_score": min_score,
        "top_k": top_k,
    }
    return final, stats
