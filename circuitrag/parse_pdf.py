"""Heuristic section splitting for papers with no HTML rendering.

Headings are guessed from short single-line blocks; if too few are found the
structure is discarded rather than reported as if it were real.
"""
from __future__ import annotations

import re

from .text import SKIP_SECTION, clean_ws


HEADING_WORDS = (
    r"(abstract|introduction|background|results?|results and discussion|discussion|conclusions?|"
    r"summary|methods?|materials and methods|methods and materials|model|models|the model|theory|"
    r"approach|outlook|supplementary (?:information|material)|appendix|acknowledge?ments?|references|bibliography)"
)
KNOWN_HEADING = re.compile(rf"^(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?{HEADING_WORDS}\s*:?$", re.I)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*|[IVX]+)\.?\s+([A-Z][^.!?]{2,90})$")
CAPTION_RE = re.compile(r"^(fig(?:ure)?\.?|table)\s*S?\d+", re.I)


def _pdf_heading(raw: str) -> tuple[int, str] | None:
    lines = [l for l in raw.strip().split("\n") if l.strip()]
    if len(lines) != 1:
        return None
    text = clean_ws(lines[0])
    if len(text.split()) > 12:
        return None
    if KNOWN_HEADING.match(text):
        return 1, re.sub(r"^(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+", "", text).rstrip(":")
    m = NUMBERED_HEADING.match(text)
    if m:
        return (2 if "." in m.group(1).strip(".") else 1), m.group(2).strip()
    return None


def parse_pdf(pdf_bytes: bytes) -> list[dict] | None:
    """Heuristic section splitting for PDFs: short single-line blocks that look like headings."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sections: list[dict] = []
    path, top, paras, skipping, n_headings = ["Front matter"], "Front matter", [], False, 0

    def flush():
        if paras and not skipping:
            sections.append({"path": list(path), "kind": "text", "paragraphs": list(paras)})
        paras.clear()

    for page in doc:
        for b in page.get_text("blocks"):
            if b[6] != 0:  # image block
                continue
            raw = b[4]
            text = clean_ws(raw.replace("-\n", ""))
            if not text:
                continue
            h = _pdf_heading(raw)
            if h:
                level, title = h
                flush()
                if re.match(r"^(references|bibliography)$", title, re.I):
                    return _finish_pdf(sections, n_headings)
                skipping = bool(SKIP_SECTION.match(title))
                n_headings += 1
                if level == 1:
                    top, path = title, [title]
                else:
                    path = [top, title]
                continue
            if skipping:
                continue
            if CAPTION_RE.match(text):
                sections.append({"path": list(path), "kind": "figure", "paragraphs": [text]})
                continue
            if len(text.split()) < 6:  # running headers, page numbers, stray labels
                continue
            paras.append(text)
    flush()
    return _finish_pdf(sections, n_headings)


def _finish_pdf(sections: list[dict], n_headings: int) -> list[dict] | None:
    if not sections:
        return None
    if n_headings >= 2:  # front matter duplicates the abstract we already have
        sections = [s for s in sections if s["path"][0] != "Front matter"] or sections
    else:
        for s in sections:
            s["path"] = ["Body"]
    return sections
