"""
crawler_browser.py — RAG crawler using Playwright via crawl4ai.

Renders JavaScript, simulates real user behaviour. Use when the fast
crawler misses content due to anti-bot protection or JS rendering.

Usage:
    python -m crawler.crawler_browser --urls-file urls.txt
    python -m crawler.crawler_browser --urls https://example.com --concurrency 8

Requirements:
    pip install crawl4ai aiofiles trafilatura lxml
    playwright install chromium
"""

import asyncio
import logging

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_configs import CacheMode

from .bfs import run_crawler
from .cli import base_arg_parser, load_urls, make_config

log = logging.getLogger("crawler")

_CRAWL4AI_CONFIG = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    simulate_user=True,
    magic=True,
    remove_consent_popups=True,
    excluded_tags=[
        "nav",
        "footer",
        "header",
        "aside",
        "script",
        "style",
        "form",
        "noscript",
    ],
    word_count_threshold=5,
    verbose=False,
)


async def main():
    p = base_arg_parser(
        "Browser crawler (crawl4ai + Playwright) — JS rendering, anti-bot stealth"
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="Parallel browser tabs (default: 30, each uses ~50-100MB RAM)",
    )
    args = p.parse_args()

    urls = load_urls(args)
    config = make_config(args, concurrency=args.concurrency)

    log.info(
        "Mode: browser (crawl4ai) | Concurrency: %d | Readability: %s",
        args.concurrency,
        config.readability,
    )

    async with AsyncWebCrawler(verbose=False) as crawler:

        async def fetch(url: str) -> bytes | None:
            try:
                result = await crawler.arun(url=url, config=_CRAWL4AI_CONFIG)
            except Exception as e:
                log.debug("  ✗ %s: %s", url, e)
                return None

            html = getattr(result, "html", "") or ""
            if not html:
                inner = getattr(result, "_results", None)
                if inner:
                    html = getattr(inner[0], "html", "") or ""
            if not html:
                return None

            return html.encode("utf-8") if isinstance(html, str) else html

        await run_crawler(urls, config, fetch)


if __name__ == "__main__":
    asyncio.run(main())
