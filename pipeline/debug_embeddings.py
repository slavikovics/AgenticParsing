"""
debug_embeddings.py — sanity check for the embedding pipeline.

Runs 3 tests and prints what's actually happening:
1. Single text embed — are vectors non-zero and normalised?
2. Query vs document similarity — does a relevant query score high?
3. Query vs random text — does an irrelevant query score low?

Usage:
    python -m pipeline_pkg.debug_embeddings
"""

import numpy as np
import torch

from .config import MAX_CHUNK_TOKENS, MODEL_ID, QUERY_INSTRUCTION
from .embedder import Embedder


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run():
    print(f"\nModel:          {MODEL_ID}")
    print(f"MAX_CHUNK_TOKENS: {MAX_CHUNK_TOKENS}")
    print(f"QUERY_INSTRUCTION prefix length: {len(QUERY_INSTRUCTION)} chars\n")

    embedder = Embedder(batch_size=1)

    # ── Test 1: basic sanity ──────────────────────────────────────────────────
    print("=" * 55)
    print("Test 1: Single embed sanity")
    vec = embedder.embed_documents(
        ["Университет принимает заявления от абитуриентов."]
    )[0]
    print(f"  Vector shape:    {vec.shape}")
    print(f"  Vector norm:     {np.linalg.norm(vec):.6f}  (should be ~1.0)")
    print(f"  First 5 values:  {vec[:5]}")
    print(f"  Non-zero count:  {np.count_nonzero(vec)} / {len(vec)}")

    # ── Test 2: relevant query should score high ──────────────────────────────
    print("\n" + "=" * 55)
    print("Test 2: Relevant query vs relevant document")
    doc = (
        "Поступление в университет. Для зачисления на первый курс необходимо "
        "предоставить документы: аттестат о среднем образовании, результаты "
        "централизованного тестирования, заявление установленного образца. "
        "Приёмная комиссия работает с 9:00 до 17:00."
    )
    query = "поступление в университет требования для абитуриентов"

    doc_vec = embedder.embed_documents([doc])[0]
    query_vec = embedder.embed_queries([query])[0]

    sim = cosine(doc_vec, query_vec)
    print(f"  Document (first 80 chars): {doc[:80]}...")
    print(f"  Query:    {query}")
    print(f"  Cosine similarity: {sim:.4f}  (expect > 0.6)")

    # Also show what query looks like with prefix vs without
    raw_query_vec = embedder.embed_documents([query])[0]  # no prefix
    sim_no_prefix = cosine(doc_vec, raw_query_vec)
    print(f"  Cosine WITHOUT prefix:     {sim_no_prefix:.4f}")

    # ── Test 3: irrelevant query should score low ─────────────────────────────
    print("\n" + "=" * 55)
    print("Test 3: Irrelevant query vs same document")
    bad_query = "рецепты приготовления борща на зиму"
    bad_vec = embedder.embed_queries([bad_query])[0]
    sim_bad = cosine(doc_vec, bad_vec)
    print(f"  Query:    {bad_query}")
    print(f"  Cosine similarity: {sim_bad:.4f}  (expect < 0.3)")

    # ── Test 4: verify pooling is finding the right token ────────────────────
    print("\n" + "=" * 55)
    print("Test 4: Pooling position check")
    texts = [
        "Краткий текст.",
        "Это немного более длинный текст для проверки паддинга и пулинга.",
    ]
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from .embedder import _last_token_pool

    tok = embedder.tokenizer
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_CHUNK_TOKENS,
        return_tensors="pt",
    )
    print(f"  Batch input_ids shape: {enc['input_ids'].shape}")
    print(f"  padding_side:          {tok.padding_side}")
    print(f"  attention_mask row 0:  {enc['attention_mask'][0].tolist()}")
    print(f"  attention_mask row 1:  {enc['attention_mask'][1].tolist()}")
    # Check which position last_token_pool would pick
    left_padding = enc["attention_mask"][:, -1].sum() == enc["attention_mask"].shape[0]
    print(f"  Detected left_padding: {left_padding}  (should be True)")
    if not left_padding:
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        print(f"  ⚠ RIGHT padding detected! Pool positions: {seq_lens.tolist()}")
        print(f"  This is likely the bug — tokenizer padding_side should be 'left'")

    print("\n" + "=" * 55)
    print("Done. If Test 2 score < 0.5, the embeddings are broken.")
    print("If Test 4 shows right padding, that's the root cause.")


if __name__ == "__main__":
    run()
