"""
urls.py — URL normalisation, domain helpers, and link extraction.
"""

from pathlib import Path
from urllib.parse import urlparse

from lxml import html as lxml_html

_NON_HTML_EXT = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".css",
    ".js",
    ".json",
    ".xml",
    ".csv",
    ".rss",
    ".atom",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".exe",
    ".dmg",
    ".apk",
    ".deb",
    ".rpm",
}


def root_domain(url: str) -> str:
    netloc = urlparse(url).netloc.split(":")[0]
    parts = netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def same_domain(url: str, base: str) -> bool:
    return root_domain(url) == root_domain(base)


def normalise(url: str) -> str:
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return p._replace(query="", fragment="", path=path).geturl()


def domain_slug(url: str) -> str:
    import re

    netloc = urlparse(url).netloc.split(":")[0].lstrip("www.")
    return re.sub(r"[^\w]", "_", netloc)


def is_html_url(url: str) -> bool:
    ext = Path(urlparse(url).path.lower().rstrip("/")).suffix
    return ext == "" or ext in (".html", ".htm") or ext not in _NON_HTML_EXT


def extract_links(html_bytes: bytes, page_url: str, base_url: str) -> list[str]:
    seen = set()
    try:
        doc = lxml_html.fromstring(html_bytes, base_url=page_url)
        doc.make_links_absolute(page_url, resolve_base_href=True)
        for _el, attr, href, _ in doc.iterlinks():
            if attr != "href":
                continue
            if not href or href.startswith(
                ("mailto:", "tel:", "javascript:", "#", "data:")
            ):
                continue
            try:
                full = normalise(href)
            except Exception:
                continue
            p = urlparse(full)
            if p.scheme not in ("http", "https"):
                continue
            if "lib" in p.netloc.lower():
                continue
            if not same_domain(full, base_url):
                continue
            if not is_html_url(full):
                continue
            seen.add(full)
    except Exception:
        pass
    return list(seen)
