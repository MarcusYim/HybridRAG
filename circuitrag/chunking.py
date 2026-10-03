"""Pack a paper's sections into embeddable chunks."""
from __future__ import annotations

import re


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _units(paragraphs: list[str], max_words: int) -> list[str]:
    """Paragraphs as packing units; oversized ones are broken into sentences, then word windows."""
    units = []
    for p in paragraphs:
        if len(p.split()) <= max_words:
            units.append(p)
            continue
        for s in SENT_SPLIT.split(p):
            w = s.split()
            for i in range(0, len(w), max_words):
                units.append(" ".join(w[i : i + max_words]))
    return units


def pack(paragraphs: list[str], max_words: int, overlap_words: int) -> list[str]:
    chunks, cur, cur_n = [], [], 0
    for u in _units(paragraphs, max_words):
        n = len(u.split())
        if cur and cur_n + n > max_words:
            chunks.append(" ".join(cur))
            carry, carry_n = [], 0  # overlap: carry trailing units that fit in the overlap budget
            for prev in reversed(cur):
                pn = len(prev.split())
                if carry_n + pn > overlap_words:
                    break
                carry.insert(0, prev)
                carry_n += pn
            cur, cur_n = carry, carry_n
        cur.append(u)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_paper(p: dict, max_words: int, overlap_words: int, min_words: int = 8) -> list[dict]:
    year = int(p["published"][:4]) if p.get("published", "")[:4].isdigit() else None
    raw = [(["Abstract"], "abstract", p["abstract"])]
    for s in p.get("sections", []):
        for text in pack(s["paragraphs"], max_words, overlap_words):
            raw.append((s["path"], s["kind"], text))
    chunks = []
    for path, kind, text in raw:
        if len(text.split()) < min_words:
            continue
        chunks.append(
            {
                "chunk_id": f"{p['id']}#{len(chunks)}",
                "paper_id": p["id"],
                "title": p["title"],
                "path": " > ".join(path),
                "kind": kind,
                "year": year,
                "text": text,
            }
        )
    return chunks


def chunk_context_text(c: dict) -> str:
    """What actually gets embedded / keyword-indexed: text plus its location in the paper."""
    label = "Figure caption" if c["kind"] == "figure" else "Section"
    return f"{c['title']}\n{label}: {c['path']}\n\n{c['text']}"
