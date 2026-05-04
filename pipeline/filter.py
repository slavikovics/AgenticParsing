"""
filter.py — cosine similarity scoring and top-K filtering.
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
    # (n_chunks, n_queries)
    scores = doc_embeddings @ query_embeddings.T
    max_scores = scores.max(axis=1)

    for chunk, score in zip(chunks, max_scores):
        chunk.similarity = float(score)

    return chunks


def filter_top_k(chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Return the top-K chunks by similarity score, sorted descending."""
    return sorted(chunks, key=lambda c: c.similarity, reverse=True)[:top_k]
