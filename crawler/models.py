"""
models.py — data classes shared across all crawler modules.
"""

import time
from dataclasses import asdict, dataclass, field


@dataclass
class Document:
    url: str
    base_url: str
    title: str
    markdown: str
    content_hash: str
    crawled_at: float = field(default_factory=time.time)
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrawlConfig:
    max_depth: int = 4
    max_pages_per_domain: int = 999_999
    concurrency: int = 50
    min_words: int = 20
    request_timeout: int = 15
    readability: bool = True
    output_dir: str = "./crawl_output"
    save_individual_files: bool = True
