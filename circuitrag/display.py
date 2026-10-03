"""Terminal rendering of search results."""
from __future__ import annotations

import os
import re
import sys

from .query import parse_query, phrase_regex
from .search import Hit
from .text import tokenize


USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def snippet(text: str, query: str, width: int = 70) -> str:
    pq = parse_query(query)
    terms = set(tokenize(pq.text)) | {w for ph in pq.phrases for w in tokenize(ph)}
    words = text.split()
    if len(words) > width and terms:
        hits = [any(t in w.lower() for t in terms) for w in words]
        best = max(range(0, len(words) - width + 1, 5), key=lambda s: sum(hits[s : s + width]))
        words = (["…"] if best else []) + words[best : best + width] + (["…"] if best + width < len(text.split()) else [])
    elif len(words) > width:
        words = words[:width] + ["…"]
    out = " ".join(words)
    for ph in pq.phrases:
        out = re.sub(phrase_regex(ph).pattern, lambda m: _c(m.group(0), "1;33"), out, flags=re.I)
    for t in sorted(terms, key=len, reverse=True):
        if len(t) > 2:
            out = re.sub(rf"(?<![\w\x1b])({re.escape(t)})", lambda m: _c(m.group(1), "1"), out, flags=re.I)
    return out


def print_hits(hits: list[Hit], query: str, show_tags: bool = True):
    if not hits:
        print("No results.")
        return
    for n, h in enumerate(hits, 1):
        c, p = h.chunk, h.paper
        print(_c(f"[{n}] {p['title']}", "1;36") + f"  ({p['id']}, {c['year']})")
        print(_c(f"    {c['kind']}: {c['path']}", "35") +
              f"   score {h.score:.3f} | dense {h.dense:.3f} | bm25 {h.keyword:.2f}")
        if show_tags:
            t = p["tags"]
            tag_str = "; ".join(f"{k}: {', '.join(v[:4])}" for k, v in t.items() if v and k != "kw")
            if tag_str:
                print(_c(f"    tags  {tag_str}", "2"))
        print("    " + snippet(c["text"], query))
        print()
