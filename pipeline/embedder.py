"""
embedder.py — GPU embedding using Qwen3-Embedding via sentence-transformers.
"""

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import MODEL_ID, QUERY_INSTRUCTION

log = logging.getLogger("pipeline")


class Embedder:
    """
    Wraps Qwen3-Embedding-8B (or any sentence-transformers model).

    Documents are embedded without any instruction prefix.
    Queries are prefixed with QUERY_INSTRUCTION as required by Qwen3.
    Flash Attention 2 is enabled automatically on RTX 30xx/40xx/50xx GPUs.
    """

    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 64):
        self.batch_size = batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading %s on %s...", model_id, device)

        model_kwargs: dict = {"torch_dtype": torch.float16}
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            log.info("  Flash Attention 2 enabled")

        self.model = SentenceTransformer(
            model_id,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
            device=device,
        )
        log.info(
            "Model ready. Embedding dim: %d",
            self.model.get_sentence_embedding_dimension(),
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed document chunks — no instruction prefix."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed queries with task instruction prefix."""
        prefixed = [QUERY_INSTRUCTION + q for q in queries]
        return self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
