"""arXiv API search and full-text retrieval."""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .parse_html import parse_latexml_html
from .parse_pdf import parse_pdf
from .text import clean_ws, safe_name

USER_AGENT = "circuitrag/0.1 (research prototype; mailto:you@example.com)"


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

DEFAULT_QUERY = (
    "(cat:q-bio.MN OR cat:q-bio.SC OR cat:q-bio.QM OR cat:physics.bio-ph OR cat:eess.SY) AND "
    '(abs:"genetic circuit" OR abs:"genetic circuits" OR abs:"gene circuit" OR abs:"gene circuits" '
    'OR abs:"synthetic biology" OR abs:"synthetic gene" OR abs:"toggle switch" OR abs:repressilator '
    'OR abs:"biomolecular circuit" OR abs:"gene regulatory circuit")'
)


def arxiv_search(query: str, max_results: int, sort_by: str = "relevance", page_size: int = 50) -> list[dict]:
    import requests

    papers, start = [], 0
    while start < max_results:
        n = min(page_size, max_results - start)
        r = requests.get(
            ARXIV_API,
            params={
                "search_query": query,
                "start": start,
                "max_results": n,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        entries = ET.fromstring(r.content).findall(f"{ATOM}entry")
        if not entries:
            break
        papers.extend(_parse_entry(e) for e in entries)
        start += len(entries)
        if len(entries) < n:
            break
        time.sleep(3)  # arXiv asks for >=3 s between API calls
    return papers


def _parse_entry(e) -> dict:
    abs_url = clean_ws(e.findtext(f"{ATOM}id"))
    arxiv_id = abs_url.rsplit("/abs/", 1)[-1]
    pdf_url = next(
        (l.get("href") for l in e.findall(f"{ATOM}link") if l.get("title") == "pdf"),
        f"https://arxiv.org/pdf/{arxiv_id}",
    )
    return {
        "id": arxiv_id,
        "title": clean_ws(e.findtext(f"{ATOM}title")),
        "abstract": clean_ws(e.findtext(f"{ATOM}summary")),
        "authors": [clean_ws(a.findtext(f"{ATOM}name")) for a in e.findall(f"{ATOM}author")],
        "published": (e.findtext(f"{ATOM}published") or "")[:10],
        "categories": [c.get("term") for c in e.findall(f"{ATOM}category")],
        "url": abs_url,
        "pdf_url": pdf_url,
    }

def fetch_fulltext(paper: dict) -> tuple[str, list[dict]]:
    import requests

    headers = {"User-Agent": USER_AGENT}
    bare_id = re.sub(r"v\d+$", "", paper["id"])
    for url in (f"https://arxiv.org/html/{paper['id']}", f"https://ar5iv.labs.arxiv.org/html/{bare_id}"):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200:
                secs = parse_latexml_html(r.text)
                if secs:
                    return "html", secs
        except Exception as exc:  # network hiccup, parse failure: try the next source
            print(f"    html failed ({url}): {exc}", file=sys.stderr)
    try:
        r = requests.get(paper["pdf_url"], headers=headers, timeout=120)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            secs = parse_pdf(r.content)
            if secs:
                return "pdf", secs
    except ImportError:
        print("    PyMuPDF not installed; skipping PDF fallback (pip install pymupdf)", file=sys.stderr)
    except Exception as exc:
        print(f"    pdf failed: {exc}", file=sys.stderr)
    return "abstract", []


def fetch_corpus(out_dir: str | Path, query: str = DEFAULT_QUERY, max_results: int = 60,
                 sort_by: str = "relevance", full_text: bool = True, refresh: bool = False,
                 delay: float = 1.0) -> list[dict]:
    """Download papers into `out_dir`, one JSON file each. Cached by arXiv ID."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Querying arXiv for up to {max_results} papers...")
    papers = arxiv_search(query, max_results, sort_by=sort_by)
    print(f"Found {len(papers)} papers.")
    for i, p in enumerate(papers, 1):
        dest = out / f"{safe_name(p['id'])}.json"
        if dest.exists() and not refresh:
            print(f"[{i}/{len(papers)}] {p['id']} cached")
            continue
        if full_text:
            p["source"], p["sections"] = fetch_fulltext(p)
            time.sleep(delay)
        else:
            p["source"], p["sections"] = "abstract", []
        dest.write_text(json.dumps(p, indent=1))
        n = sum(len(sec["paragraphs"]) for sec in p["sections"])
        print(f"[{i}/{len(papers)}] {p['id']} via {p['source']:8s} "
              f"{len(p['sections']):3d} sections {n:4d} paras  {p['title'][:60]}")
    return papers
