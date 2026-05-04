"""
text.py — HTML content extraction, title detection, filename generation.
"""

import hashlib
import re
from urllib.parse import urlparse

import trafilatura


def extract_text(html_bytes: bytes, url: str, readability: bool) -> str:
    """
    Extract main content from raw HTML bytes using trafilatura.

    readability=True  → precision mode: strips nav/footer/ads aggressively
    readability=False → recall mode: keeps more content, less filtering
    """
    text = trafilatura.extract(
        html_bytes,
        url=url,
        include_formatting=True,  # preserve headings/bold/lists as markdown
        include_links=False,  # skip hyperlinks — noise for RAG
        include_tables=True,
        include_images=False,
        favor_precision=readability,
        favor_recall=not readability,
        no_fallback=False,
        deduplicate=True,
    )
    return (text or "").strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def make_title(text: str, url: str) -> str:
    """Extract first H1-H3 from markdown text, fall back to URL path."""
    m = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    return (
        m.group(1).strip()
        if m
        else (urlparse(url).path.strip("/") or urlparse(url).netloc)
    )


def safe_filename(url: str) -> str:
    """Generate a unique, filesystem-safe .md filename from a URL."""
    import re as _re

    p = urlparse(url)
    raw = p.netloc.lstrip("www.") + p.path
    slug = _re.sub(r"[^\w\-]", "_", raw)
    slug = _re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = _re.sub(r"[^\w]", "_", p.netloc) or "index"
    return slug[:120] + ".md"
