"""
pipeline.py — orchestrator and CLI entrypoint.

Reads:  crawl_output/{domain}/collection.jsonl
Writes: crawl_output/{domain}/chunks.jsonl           (all chunks + scores)
        crawl_output/{domain}/chunks_filtered.jsonl  (top-K by similarity)

Usage:
    python -m pipeline_pkg.pipeline --input ./crawl_output
    python -m pipeline_pkg.pipeline --input ./crawl_output --domain grsu_by
    python -m pipeline_pkg.pipeline --input ./crawl_output --top-k 5000

Requirements:
    pip install sentence-transformers>=2.7.0 transformers>=4.51.0 torch numpy tqdm
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .chunker import chunk_collection
from .config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS, MODEL_ID, RELEVANCE_QUERIES
from .embedder import Embedder
from .filter import filter_top_k, score_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=__import__("sys").stdout,
    force=True,
)
log = logging.getLogger("pipeline")


def process_domain(
    domain_dir: Path,
    embedder: Embedder,
    query_embeddings,
    top_k: int,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_words: int,
    max_chunk_words: int,
):
    collection_path = domain_dir / "collection.jsonl"
    if not collection_path.exists():
        log.warning("No collection.jsonl in %s — skipping", domain_dir)
        return

    log.info("━━ %s ━━", domain_dir.name)
    t0 = time.time()

    # 1. Chunk
    log.info("Chunking...")
    chunks = chunk_collection(
        collection_path, chunk_tokens, overlap_tokens, min_chunk_words, max_chunk_words
    )
    if not chunks:
        log.warning("No chunks produced for %s", domain_dir.name)
        return

    # 2. Embed
    log.info("Embedding %d chunks...", len(chunks))
    doc_embeddings = embedder.embed_documents([c.text for c in chunks])

    # 3. Score
    log.info("Scoring against %d queries...", len(RELEVANCE_QUERIES))
    chunks = score_chunks(chunks, doc_embeddings, query_embeddings)

    # 4. Save all chunks with scores
    chunks_path = domain_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    log.info("Saved %d chunks → %s", len(chunks), chunks_path)

    # 5. Filter top-K and save
    filtered = filter_top_k(chunks, top_k)
    filtered_path = domain_dir / "chunks_filtered.jsonl"
    with open(filtered_path, "w", encoding="utf-8") as f:
        for chunk in filtered:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    top_score = filtered[0].similarity if filtered else 0
    bot_score = filtered[-1].similarity if filtered else 0
    log.info(
        "Filtered → %d chunks in %.1fs | scores: %.3f – %.3f → %s",
        len(filtered),
        elapsed,
        top_score,
        bot_score,
        filtered_path,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Chunk + embed + filter crawled collections"
    )
    p.add_argument("--input", default="./crawl_output")
    p.add_argument(
        "--domain", default=None, help="Process only this domain slug, e.g. grsu_by"
    )
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument(
        "--chunk-size", type=int, default=400, help="Target chunk size in tokens"
    )
    p.add_argument(
        "--overlap", type=int, default=50, help="Overlap between chunks in tokens"
    )
    p.add_argument("--min-chunk-words", type=int, default=CHUNK_MIN_WORDS)
    p.add_argument("--max-chunk-words", type=int, default=CHUNK_MAX_WORDS)
    p.add_argument(
        "--batch-size", type=int, default=64, help="GPU embedding batch size"
    )
    p.add_argument("--model", default=MODEL_ID)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.input)

    if not root.exists():
        log.error("Input directory not found: %s", root)
        return

    domain_dirs = (
        [root / args.domain]
        if args.domain
        else [d for d in sorted(root.iterdir()) if d.is_dir()]
    )
    if not domain_dirs:
        log.error("No domain directories found in %s", root)
        return

    log.info("Domains: %s", [d.name for d in domain_dirs])
    log.info(
        "Top-K: %d | Chunk: %d tokens | Overlap: %d | Words: %d–%d",
        args.top_k,
        args.chunk_size,
        args.overlap,
        args.min_chunk_words,
        args.max_chunk_words,
    )

    embedder = Embedder(args.model, batch_size=args.batch_size)

    log.info("Embedding %d relevance queries...", len(RELEVANCE_QUERIES))
    query_embeddings = embedder.embed_queries(RELEVANCE_QUERIES)

    for domain_dir in domain_dirs:
        try:
            process_domain(
                domain_dir,
                embedder,
                query_embeddings,
                top_k=args.top_k,
                chunk_tokens=args.chunk_size,
                overlap_tokens=args.overlap,
                min_chunk_words=args.min_chunk_words,
                max_chunk_words=args.max_chunk_words,
            )
        except Exception as e:
            log.error("Failed on %s: %s", domain_dir.name, e, exc_info=True)

    log.info("All done.")


if __name__ == "__main__":
    main()
