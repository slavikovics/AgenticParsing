"""
crawler.py — fast RAG crawler using plain HTTP (aiohttp).

No browser, no JavaScript. Best for static/server-rendered sites.

Usage:
    python -m crawler.crawler --urls-file urls.txt
    python -m crawler.crawler --urls https://example.com --concurrency 30

Requirements:
    pip install aiohttp aiofiles trafilatura lxml
"""

import asyncio
import logging

import aiohttp

from .bfs import run_crawler
from .cli import base_arg_parser, load_urls, make_config

log = logging.getLogger("crawler")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/ragbot)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ru,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


async def main():
    p = base_arg_parser("Fast aiohttp crawler — no browser, maximum speed")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--timeout", type=int, default=15)
    args = p.parse_args()

    urls = load_urls(args)
    config = make_config(args, concurrency=args.concurrency)
    config.request_timeout = args.timeout

    log.info(
        "Mode: aiohttp | Concurrency: %d | Readability: %s",
        args.concurrency,
        config.readability,
    )

    connector = aiohttp.TCPConnector(
        limit=0, ttl_dns_cache=300, enable_cleanup_closed=True
    )
    session = aiohttp.ClientSession(
        headers=HEADERS,
        connector=connector,
        cookie_jar=aiohttp.CookieJar(),
    )

    async def fetch(url: str) -> bytes | None:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=config.request_timeout),
                allow_redirects=True,
                ssl=False,
            ) as resp:
                if resp.status != 200:
                    log.debug("  ✗ HTTP %d: %s", resp.status, url)
                    return None
                ct = resp.content_type or ""
                if "html" not in ct and ct != "":
                    log.debug("  ~ non-html (%s): %s", ct, url)
                    return None
                return await resp.read()
        except asyncio.TimeoutError:
            log.debug("  ✗ timeout: %s", url)
        except aiohttp.ClientError as e:
            log.debug("  ✗ %s: %s", type(e).__name__, url)
        except Exception as e:
            log.debug("  ✗ %s: %s", e, url)
        return None

    try:
        await run_crawler(urls, config, fetch)
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
