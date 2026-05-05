"""
qdrant_upload.py — upload chunks_filtered.jsonl into Qdrant.

One domain = one collection. Collection name = domain slug (e.g. grsu_by).
Uses gRPC for fast bulk upload. Creates collection if it doesn't exist,
recreates it if vector dimensions changed (e.g. model was swapped).

Usage:
    # Upload all domains
    python -m pipeline_pkg.qdrant_upload --input ./crawl_output

    # Upload one domain
    python -m pipeline_pkg.qdrant_upload --input ./crawl_output --domain grsu_by

    # Custom Qdrant host
    python -m pipeline_pkg.qdrant_upload --input ./crawl_output --host localhost --port 6334

    # Upload chunks.jsonl instead of filtered version
    python -m pipeline_pkg.qdrant_upload --input ./crawl_output --all-chunks

Requirements:
    pip install qdrant-client
"""

import argparse
import json
import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

log = logging.getLogger("pipeline")
UPLOAD_BATCH_SIZE = 2_000


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return chunks


def chunk_to_point(chunk: dict, idx: int) -> PointStruct | None:
    vector = chunk.get("embedding")
    if not vector:
        return None
    return PointStruct(
        id=idx,
        vector=vector,
        payload={
            "chunk_id": chunk.get("chunk_id", ""),
            "doc_url": chunk.get("doc_url", ""),
            "doc_base_url": chunk.get("doc_base_url", ""),
            "doc_title": chunk.get("doc_title", ""),
            "text": chunk.get("text", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 1),
            "word_count": chunk.get("word_count", 0),
            "similarity": chunk.get("similarity", 0.0),
        },
    )


def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    if client.collection_exists(name):
        info = client.get_collection(name)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            log.warning(
                "Collection %s has dim=%d but chunks have dim=%d — recreating",
                name,
                existing_size,
                vector_size,
            )
            client.delete_collection(name)
        else:
            log.info("Collection %s already exists (%d dims)", name, vector_size)
            return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
    )
    log.info("Created collection %s (dim=%d)", name, vector_size)


def upload_domain(
    client: QdrantClient,
    domain_dir: Path,
    use_all_chunks: bool,
) -> int:
    filename = "chunks.jsonl" if use_all_chunks else "chunks_filtered.jsonl"
    chunks_path = domain_dir / filename

    if not chunks_path.exists():
        log.warning("No %s in %s — skipping", filename, domain_dir)
        return 0

    chunks = load_chunks(chunks_path)
    if not chunks:
        log.warning("Empty file: %s", chunks_path)
        return 0

    # Check embeddings are present
    if "embedding" not in chunks[0]:
        log.error(
            "%s chunks have no 'embedding' field.\n"
            "Run the embedder step first to add embeddings to chunks.",
            domain_dir.name,
        )
        return 0

    vector_size = len(chunks[0]["embedding"])
    collection_name = domain_dir.name  # e.g. grsu_by

    ensure_collection(client, collection_name, vector_size)

    # Build points
    points = []
    skipped = 0
    for idx, chunk in enumerate(chunks):
        pt = chunk_to_point(chunk, idx)
        if pt is None:
            skipped += 1
            continue
        points.append(pt)

    if skipped:
        log.warning("Skipped %d chunks with missing embeddings", skipped)

    # Upload in batches
    t0 = time.time()
    total = len(points)
    uploaded = 0

    for i in range(0, total, UPLOAD_BATCH_SIZE):
        batch = points[i : i + UPLOAD_BATCH_SIZE]
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
        )
        uploaded += len(batch)
        log.info("  %s: %d / %d uploaded", collection_name, uploaded, total)

    elapsed = time.time() - t0
    log.info(
        "◀ %s — %d points in %.1fs (%.0f pts/s)",
        collection_name,
        total,
        elapsed,
        total / max(elapsed, 0.01),
    )
    return total


def parse_args():
    p = argparse.ArgumentParser(description="Upload filtered chunks to Qdrant")
    p.add_argument("--input", default="./crawl_output")
    p.add_argument("--domain", default=None, help="Process only this domain slug")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=6334, help="gRPC port (default: 6334)")
    p.add_argument(
        "--all-chunks",
        action="store_true",
        help="Upload chunks.jsonl instead of chunks_filtered.jsonl",
    )
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

    log.info("Connecting to Qdrant at %s:%d (gRPC)", args.host, args.port)
    client = QdrantClient(host=args.host, grpc_port=args.port, prefer_grpc=True)

    total_uploaded = 0
    for domain_dir in domain_dirs:
        try:
            n = upload_domain(client, domain_dir, use_all_chunks=args.all_chunks)
            total_uploaded += n
        except Exception as e:
            log.error("Failed on %s: %s", domain_dir.name, e, exc_info=True)

    log.info("Done — %d total points uploaded", total_uploaded)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    main()
