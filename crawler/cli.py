"""
cli.py — shared CLI argument parser and config helpers.
"""

from argparse import ArgumentParser
from pathlib import Path

from .models import CrawlConfig


def base_arg_parser(description: str) -> ArgumentParser:
    p = ArgumentParser(description=description)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--urls", nargs="+", metavar="URL")
    src.add_argument("--urls-file", metavar="FILE")
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-pages", type=int, default=999_999)
    p.add_argument("--min-words", type=int, default=20)
    p.add_argument("--output", default="./crawl_output")
    p.add_argument("--no-individual-files", action="store_true")
    p.add_argument(
        "--no-readability",
        action="store_true",
        help="Disable trafilatura precision mode (faster, less clean)",
    )
    return p


def load_urls(args) -> list[str]:
    if args.urls_file:
        return [
            l.strip()
            for l in Path(args.urls_file).read_text().splitlines()
            if l.strip()
        ]
    return args.urls


def make_config(args, concurrency: int) -> CrawlConfig:
    return CrawlConfig(
        max_depth=args.max_depth,
        max_pages_per_domain=args.max_pages,
        concurrency=concurrency,
        min_words=args.min_words,
        readability=not args.no_readability,
        output_dir=args.output,
        save_individual_files=not args.no_individual_files,
    )
