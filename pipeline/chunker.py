"""
chunker.py — text splitting and chunk size normalisation.

Splits documents at paragraph → sentence → word boundaries,
then enforces min/max word count bounds on the resulting chunks.
"""

import json
import logging
import re
from pathlib import Path

from .config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS
from .models import Chunk

log = logging.getLogger("pipeline")


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35)


def _tail_tokens(parts: list[str], token_budget: int) -> str:
    result = []
    budget = 0
    for part in reversed(parts):
        t = estimate_tokens(part)
        if budget + t > token_budget:
            break
        result.insert(0, part)
        budget += t
    return "\n\n".join(result)


def split_into_chunks(
    text: str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
    min_words: int = 30,
) -> list[str]:
    """
    Split text into overlapping chunks at natural boundaries:
    1. Paragraph breaks (double newline or markdown headers)
    2. Sentence boundaries
    3. Hard split if a single sentence exceeds chunk size

    Overlap carries the tail of each chunk into the next,
    so a relevant sentence near a boundary appears in both.
    """
    paragraphs = re.split(r"\n{2,}|(?=\n#{1,3} )", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        if current_tokens + para_tokens <= chunk_tokens:
            current.append(para)
            current_tokens += para_tokens

        elif para_tokens > chunk_tokens:
            if current:
                chunks.append("\n\n".join(current))
                overlap = _tail_tokens(current, overlap_tokens)
                current = [overlap] if overlap else []
                current_tokens = estimate_tokens(overlap)

            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = estimate_tokens(sent)
                if current_tokens + sent_tokens <= chunk_tokens:
                    current.append(sent)
                    current_tokens += sent_tokens
                else:
                    if current:
                        chunks.append(" ".join(current))
                    overlap = _tail_tokens(current, overlap_tokens)
                    current = [overlap, sent] if overlap else [sent]
                    current_tokens = estimate_tokens(" ".join(current))
        else:
            if current:
                chunks.append("\n\n".join(current))
                overlap = _tail_tokens(current, overlap_tokens)
                current = [overlap, para] if overlap else [para]
                current_tokens = estimate_tokens("\n\n".join(current))
            else:
                current = [para]
                current_tokens = para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if len(c.split()) >= min_words]


def normalise_chunk_sizes(
    parts: list[str],
    max_words: int = CHUNK_MAX_WORDS,
    min_words: int = CHUNK_MIN_WORDS,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Post-process chunks to enforce size bounds:
    - Drop chunks below min_words (headings, stubs, fragments)
    - Re-split chunks above max_words at sentence boundaries
    """
    result = []
    for part in parts:
        words = len(part.split())
        if words < min_words:
            continue
        if words <= max_words:
            result.append(part)
            continue
        # Re-split oversized chunk by sentence
        sentences = re.split(r"(?<=[.!?])\s+", part)
        current: list[str] = []
        current_words = 0
        for sent in sentences:
            sw = len(sent.split())
            if current_words + sw > max_words and current:
                result.append(" ".join(current))
                overlap_sent = current[-1] if current else ""
                current = [overlap_sent, sent] if overlap_sent else [sent]
                current_words = len(" ".join(current).split())
            else:
                current.append(sent)
                current_words += sw
        if current and len(" ".join(current).split()) >= min_words:
            result.append(" ".join(current))
    return result


def chunk_collection(
    collection_path: Path,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
    min_chunk_words: int = CHUNK_MIN_WORDS,
    max_chunk_words: int = CHUNK_MAX_WORDS,
) -> list[Chunk]:
    """Read a collection.jsonl and return all normalised chunks."""
    chunks: list[Chunk] = []
    doc_count = 0
    skipped = 0

    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("markdown", "")
            if not text:
                skipped += 1
                continue

            doc_count += 1
            parts = split_into_chunks(text, chunk_tokens, overlap_tokens)
            parts = normalise_chunk_sizes(
                parts, max_chunk_words, min_chunk_words, overlap_tokens
            )

            for i, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['content_hash']}_{i}",
                        doc_url=doc["url"],
                        doc_base_url=doc["base_url"],
                        doc_title=doc.get("title", ""),
                        text=part,
                        chunk_index=i,
                        total_chunks=len(parts),
                        word_count=len(part.split()),
                    )
                )

    log.info(
        "  Chunked %d docs → %d chunks (%d empty skipped)",
        doc_count,
        len(chunks),
        skipped,
    )
    return chunks
