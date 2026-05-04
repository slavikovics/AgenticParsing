# Agentic parsing
Parsing & crawling websites to extract information for RAG.

## Structure
```text
crawler/
    models.py          ← Document, CrawlConfig
    urls.py            ← normalise, root_domain, extract_links
    text.py            ← extract_text, content_hash, make_title, safe_filename
    writer.py          ← OutputWriter (per-domain JSONL + .md files)
    bfs.py             ← DomainState, run_crawler (shared BFS engine)
    cli.py             ← base_arg_parser, load_urls, make_config
    crawler.py         ← aiohttp entrypoint  (python -m crawler.crawler)
    crawler_browser.py ← crawl4ai entrypoint (python -m crawler.crawler_browser)

pipeline_pkg/
    config.py          ← MODEL_ID, RELEVANCE_QUERIES, chunk size bounds  ← edit this
    models.py          ← Chunk dataclass
    chunker.py         ← split_into_chunks, normalise_chunk_sizes, chunk_collection
    embedder.py        ← Embedder (Qwen3, Flash Attention, batch encode)
    filter.py          ← score_chunks, filter_top_k
    pipeline.py        ← process_domain orchestrator + CLI entrypoint
```

## How to run
```bash
python -m crawler.crawler --urls-file urls.txt
python -m crawler.crawler_browser --urls-file urls.txt
python -m pipeline_pkg.pipeline --input ./crawl_output --top-k 5000
```
