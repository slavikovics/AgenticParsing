"""
models.py — data classes for the pipeline.
"""

from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    chunk_id: str  # "{doc_content_hash}_{chunk_index}"
    doc_url: str
    doc_base_url: str
    doc_title: str
    text: str
    chunk_index: int
    total_chunks: int
    word_count: int
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
