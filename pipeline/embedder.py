"""
embedder.py — GPU embedding using Qwen3-Embedding.

Loads via transformers directly (not SentenceTransformer wrapper)
so quantization_config is properly applied to the model weights.

Memory guide for 16GB VRAM:
    8B  float16  → ~14GB  (too tight)
    8B  int4     → ~5GB   ← default when bitsandbytes installed
    4B  float16  → ~6GB
    0.6B float16 → ~1.5GB
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import MAX_CHUNK_TOKENS, MODEL_ID, QUERY_INSTRUCTION

log = logging.getLogger("pipeline")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _free_vram_gb() -> float:
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    return 0.0


def _last_token_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Last-token (EOS) pooling — required by Qwen3-Embedding.
    With left-padding (padding_side="left"), the last token is always the EOS.
    With right-padding, find the actual last non-pad token per sequence.
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths,
        ]


class Embedder:
    def __init__(self, model_id: str = MODEL_ID, batch_size: int = 8):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        free_gb = _free_vram_gb()
        log.info(
            "Loading %s | device=%s | free VRAM=%.1fGB", model_id, self.device, free_gb
        )

        # ── Build model kwargs ────────────────────────────────────────────────
        model_kwargs: dict = {}

        # 4-bit quantization — load BEFORE moving weights to GPU
        use_quantization = False
        if self.device == "cuda":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "cuda"
                use_quantization = True
                log.info("  4-bit quantization enabled (bitsandbytes)")
            except ImportError:
                log.warning("  bitsandbytes not found — loading in float16 (may OOM)")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "cuda"

        # Flash attention
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            for pkg in ("flash_attn_3", "flash_attn"):
                try:
                    __import__(pkg)
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    log.info("  %s enabled", pkg)
                    break
                except ImportError:
                    continue

        # ── Load tokenizer + model ────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs)

        if not use_quantization:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.dim = self.model.config.hidden_size

        free_after = _free_vram_gb()
        log.info(
            "Model ready | VRAM free: %.1fGB | batch_size=%d | dim=%d",
            free_after,
            self.batch_size,
            self.dim,
        )

        if self.device == "cuda" and free_after < 0.5 and self.batch_size > 4:
            self.batch_size = 4
            log.warning("Low VRAM — reduced batch_size to %d", self.batch_size)

    @torch.no_grad()
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_CHUNK_TOKENS,  # matches chunker target — no wasted padding
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**enc)
        emb = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=-1)
        return emb.cpu().float().numpy()

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        all_embeddings = []
        batch_size = self.batch_size

        if show_progress:
            from tqdm import tqdm

            batches = list(range(0, len(texts), batch_size))
            iterator = tqdm(batches, desc="Embedding", unit="batch")
        else:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i : i + batch_size]
            while True:
                try:
                    embeddings = self._encode_batch(batch)
                    all_embeddings.append(embeddings)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    new_bs = batch_size // 2
                    if new_bs < 1:
                        raise RuntimeError(
                            "CUDA OOM at batch_size=1.\n"
                            "  → Use a smaller model: --model Qwen/Qwen3-Embedding-4B\n"
                            f"  → Free VRAM: {_free_vram_gb():.1f}GB available"
                        )
                    log.warning("OOM — reducing batch_size %d → %d", batch_size, new_bs)
                    batch_size = new_bs
                    batch = texts[i : i + batch_size]  # retry smaller slice
                    if show_progress:
                        iterator = tqdm(
                            range(i, len(texts), batch_size),
                            desc="Embedding",
                            unit="batch",
                            initial=1,
                        )

        return np.vstack(all_embeddings)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, show_progress=True)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        prefixed = [QUERY_INSTRUCTION + q for q in queries]
        return self._encode(prefixed, show_progress=False)
