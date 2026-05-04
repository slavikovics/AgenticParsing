"""
writer.py — OutputWriter: saves documents to per-domain JSONL + markdown files.
"""

import asyncio
import json
import logging
from pathlib import Path

import aiofiles

from .models import CrawlConfig, Document
from .text import safe_filename
from .urls import domain_slug

log = logging.getLogger("crawler")


class OutputWriter:
    """
    Writes crawled documents immediately to disk as they arrive.

    Output layout:
        {output_dir}/
            grsu_by/
                collection.jsonl     ← one JSON doc per line
                pages/
                    grsu_by_about.md
                    ...
            bsu_by/
                collection.jsonl
                pages/
                    ...
    """

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.root = Path(config.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._domains: dict[str, dict] = {}
        self._init_lock = asyncio.Lock()
        self.total_count = 0
        self._total_lock = asyncio.Lock()

    async def _get_domain_state(self, base_url: str) -> dict:
        slug = domain_slug(base_url)
        async with self._init_lock:
            if slug not in self._domains:
                domain_dir = self.root / slug
                domain_dir.mkdir(parents=True, exist_ok=True)
                if self.config.save_individual_files:
                    (domain_dir / "pages").mkdir(exist_ok=True)
                collection_path = domain_dir / "collection.jsonl"
                f = await aiofiles.open(collection_path, "w", encoding="utf-8")
                log.info("Writing %s → %s", slug, collection_path.resolve())
                self._domains[slug] = {
                    "file": f,
                    "lock": asyncio.Lock(),
                    "count": 0,
                    "dir": domain_dir,
                }
        return self._domains[slug]

    async def close(self):
        for state in self._domains.values():
            await state["file"].close()

    async def write(self, doc: Document):
        state = await self._get_domain_state(doc.base_url)
        async with state["lock"]:
            await state["file"].write(
                json.dumps(doc.to_dict(), ensure_ascii=False) + "\n"
            )
            await state["file"].flush()
            state["count"] += 1
        async with self._total_lock:
            self.total_count += 1
        if self.config.save_individual_files:
            fname = state["dir"] / "pages" / safe_filename(doc.url)
            async with aiofiles.open(fname, "w", encoding="utf-8") as f:
                await f.write(
                    f"# {doc.title}\n\n"
                    f"**URL:** {doc.url}\n"
                    f"**Words:** {doc.word_count}\n\n"
                    f"---\n\n"
                    f"{doc.markdown}\n"
                )

    def domain_count(self, base_url: str) -> int:
        return self._domains.get(domain_slug(base_url), {}).get("count", 0)
