"""
bfs.py — BFS orchestrator shared by all crawler backends.

The only thing that differs between fast/browser crawlers is the fetch function.
Everything else — queue management, deduplication, content extraction, saving — lives here.
"""

import asyncio
import logging
import time
from urllib.parse import urlparse

from .models import CrawlConfig, Document
from .text import content_hash, extract_text, make_title
from .urls import extract_links, normalise
from .writer import OutputWriter

log = logging.getLogger("crawler")


class DomainState:
    """Tracks visited URLs and seen content hashes for one domain."""

    def __init__(self, base_url: str):
        self.base_url = normalise(base_url)
        self.visited: set[str] = {self.base_url}
        self.seen_hashes: set[str] = set()


async def run_crawler(
    urls: list[str],
    config: CrawlConfig,
    fetch_fn,
):
    """
    Generic BFS orchestrator. Accepts any async fetch function:

        async def fetch(url: str) -> bytes | None

    Returns raw HTML bytes or None to skip the page.
    All crawlers (aiohttp, browser) plug in here via fetch_fn.
    """
    writer = OutputWriter(config)
    queue: asyncio.Queue = asyncio.Queue()
    for url in urls:
        state = DomainState(url)
        queue.put_nowait((normalise(url), 0, state))
        log.info("▶ queued %s", urlparse(url).netloc)

    async def process(url: str, depth: int, state: DomainState):
        html_bytes = await fetch_fn(url)
        if html_bytes is None:
            return

        # ── Link discovery ────────────────────────────────────────────────
        if depth < config.max_depth:
            new_links = [
                link
                for link in extract_links(html_bytes, url, state.base_url)
                if link not in state.visited
                and len(state.visited) < config.max_pages_per_domain
            ]
            for link in new_links:
                state.visited.add(link)
                queue.put_nowait((link, depth + 1, state))

        # ── Content extraction ────────────────────────────────────────────
        text = extract_text(html_bytes, url, config.readability)
        if not text:
            return

        word_count = len(text.split())
        if word_count < config.min_words:
            return

        h = content_hash(text)
        if h in state.seen_hashes:
            return
        state.seen_hashes.add(h)

        doc = Document(
            url=url,
            base_url=state.base_url,
            title=make_title(text, url),
            markdown=text,
            content_hash=h,
            word_count=word_count,
        )
        await writer.write(doc)
        log.info(
            "  ✓ [%d] %s  depth:%d q:%d (%d words)",
            writer.total_count,
            url,
            depth,
            queue.qsize(),
            word_count,
        )

    async def worker():
        while True:
            try:
                url, depth, state = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                break
            try:
                await process(url, depth, state)
            finally:
                queue.task_done()

    t0 = time.time()
    try:
        await asyncio.gather(
            *[asyncio.create_task(worker()) for _ in range(config.concurrency)]
        )
    finally:
        await writer.close()

    elapsed = time.time() - t0
    log.info(
        "Done — %d documents in %.1fs → %s", writer.total_count, elapsed, writer.root
    )
