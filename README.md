# AgenticParsing

Offline data preparation system for the university RAG knowledge base.  
Crawls university websites → extracts content → chunks → embeds → filters → uploads to Qdrant.

---

## Overview

```
crawler/          HTTP or browser-based web crawler
pipeline/     Chunking, embedding, filtering, Qdrant upload
```

The two steps are fully decoupled. Run the crawler first, then run the pipeline against the output. You can re-run the pipeline with different settings without re-crawling.

---

## Prerequisites

```bash
pip install aiohttp aiofiles trafilatura lxml          # crawler (HTTP mode)
pip install crawl4ai && playwright install chromium    # crawler (browser mode, optional)
pip install torch sentence-transformers transformers bitsandbytes tqdm qdrant-client
```

GPU with 16GB VRAM recommended for the pipeline. The embedder auto-detects CUDA and applies 4-bit quantization via bitsandbytes to fit Qwen3-Embedding-8B into memory.

---

## Step 1 — Crawl

### HTTP crawler (fast, no JavaScript)
Best for most university sites. Runs 50 concurrent requests per domain.

```bash
python -m crawler.crawler --urls https://www.grsu.by/
python -m crawler.crawler --urls-file urls.txt
python -m crawler.crawler --urls https://bsu.by --output ./crawl_output --concurrency 30
```

### Browser crawler (JavaScript, anti-bot)
Use when the HTTP crawler gets blocked or misses JS-rendered content.
Each tab uses ~100MB RAM; default concurrency is 5.

```bash
python -m crawler.crawler_browser --urls https://example.by/
python -m crawler.crawler_browser --urls-file urls.txt --concurrency 8
```

### Options (both crawlers)

| Flag | Default | Description |
|---|---|---|
| `--urls` | — | One or more base URLs |
| `--urls-file` | — | Text file, one URL per line |
| `--max-depth` | 4 | Link depth to follow |
| `--max-pages` | unlimited | Max pages per domain |
| `--concurrency` | 50 / 5 | Parallel requests |
| `--min-words` | 20 | Drop pages shorter than this |
| `--output` | `./crawl_output` | Output directory |
| `--no-individual-files` | off | Skip per-page `.md` files |
| `--no-readability` | off | Disable trafilatura precision mode |

### Output layout

```
crawl_output/
    grsu_by/
        collection.jsonl       ← one JSON document per line
        pages/
            grsu_by_about.md
            grsu_by_faculties.md
            ...
    bsu_by/
        collection.jsonl
        pages/
            ...
```

Each document in `collection.jsonl` has:

```json
{
  "url": "https://www.grsu.by/about/",
  "base_url": "https://www.grsu.by/",
  "title": "О университете",
  "markdown": "...",
  "content_hash": "a3f1c2d4",
  "word_count": 312,
  "crawled_at": 1714900000.0
}
```

### Notes

- URLs are normalised — query strings and fragments are stripped. `/page?lang=ru` and `/page?lang=en` are treated as the same page and crawled once.
- Library subdomains (`lib.*`) are automatically skipped.
- Non-HTML resources (PDFs, images, JS, CSS, etc.) are never enqueued.
- Duplicate content is detected by SHA-256 hash and skipped.

---

## Step 2 — Pipeline

Reads `collection.jsonl` files, chunks documents, embeds with Qwen3-Embedding-8B, scores against relevance queries, saves filtered chunks, and uploads to Qdrant.

### Run

```bash
# Process all domains in crawl_output/
python -m pipeline.pipeline --input ./crawl_output

# Process one domain only
python -m pipeline.pipeline --input ./crawl_output --domain grsu_by

# Upload to Qdrant after pipeline
python -m pipeline.qdrant_upload --input ./crawl_output
python -m pipeline.qdrant_upload --input ./crawl_output --domain grsu_by
```

### Pipeline options

| Flag | Default | Description |
|---|---|---|
| `--input` | `./crawl_output` | Root directory from crawler |
| `--domain` | all | Process only this domain slug |
| `--min-score` | 0.65 | Cosine similarity threshold |
| `--top-k` | unlimited | Max chunks per domain after threshold |
| `--chunk-size` | 400 | Target chunk size in tokens |
| `--overlap` | 50 | Overlap between chunks in tokens |
| `--min-chunk-words` | 40 | Drop chunks shorter than this |
| `--max-chunk-words` | 296 | Re-split chunks longer than this |
| `--batch-size` | 8 | GPU embedding batch size |
| `--model` | `Qwen/Qwen3-Embedding-8B` | HuggingFace model ID |

### Upload options

| Flag | Default | Description |
|---|---|---|
| `--input` | `./crawl_output` | Same directory as pipeline |
| `--domain` | all | Upload only this domain |
| `--host` | `localhost` | Qdrant host |
| `--port` | 6334 | Qdrant gRPC port |
| `--all-chunks` | off | Upload `chunks.jsonl` instead of filtered |

### Output per domain

```
crawl_output/grsu_by/
    collection.jsonl         ← raw crawled pages (from crawler)
    chunks.jsonl             ← all chunks with similarity scores + embeddings
    chunks_filtered.jsonl    ← top chunks above min-score threshold
```

### Relevance queries

Chunks are scored against 10 pre-defined queries in `pipeline/config.py`. Each chunk keeps its highest score. Edit the `RELEVANCE_QUERIES` list to change what counts as "relevant for applicants".

### Chunk sizing

All chunk and token sizes are derived from a single constant in `pipeline/config.py`:

```python
CHUNK_TARGET_TOKENS = 400   # target chunk size
MAX_CHUNK_TOKENS    = 640   # embedder max_length (must be >= CHUNK_TARGET_TOKENS * 1.35)
```

The 1.35 factor accounts for Slavic languages tokenising at roughly 1.35 tokens per word. Changing `CHUNK_TARGET_TOKENS` automatically propagates to the embedder.

### Debugging embeddings

If similarity scores are unexpectedly low or zero, run the diagnostic:

```bash
python -m pipeline.debug_embeddings
```

This checks vector normalisation, relevant vs irrelevant similarity, and padding direction — the three most common failure modes.

### VRAM guide

| Model | Quantization | VRAM | Batch size |
|---|---|---|---|
| Qwen3-Embedding-8B | int4 (bitsandbytes) | ~5GB | 8–16 |
| Qwen3-Embedding-4B | float16 | ~6GB | 32 |
| Qwen3-Embedding-0.6B | float16 | ~1.5GB | 128 |

Use `--model Qwen/Qwen3-Embedding-4B` or `--model Qwen/Qwen3-Embedding-0.6B` if you hit OOM.  
The embedder auto-halves batch size on CUDA OOM and retries.

---

## Full example — one university

```bash
# 1. Crawl
python -m crawler.crawler --urls https://www.grsu.by/ --output ./crawl_output

# 2. Embed and filter
python -m pipeline.pipeline --input ./crawl_output --domain grsu_by --min-score 0.60

# 3. Upload to Qdrant
python -m pipeline.qdrant_upload --input ./crawl_output --domain grsu_by

# 4. Verify
curl http://localhost:6333/collections/grsu_by
```

## Full example — all universities from a file

```bash
# urls.txt contains one URL per line
python -m crawler.crawler --urls-file urls.txt --output ./crawl_output
python -m pipeline.pipeline --input ./crawl_output --min-score 0.65
python -m pipeline.qdrant_upload --input ./crawl_output
```
