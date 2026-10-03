"""Split an arXiv/ar5iv LaTeXML HTML rendering into a section tree.

This is the preferred source: it carries the author's own sectioning, so chunks
line up with real subsections instead of guessed boundaries.
"""
from __future__ import annotations

import re

from .text import SKIP_SECTION, clean_ws


def _classes(el) -> list[str]:
    return el.get("class") or []


def _section_ancestor(el):
    """Nearest enclosing <section> that is a real section (not a \\paragraph block)."""
    for parent in el.parents:
        if parent.name == "section" and "ltx_paragraph" not in _classes(parent):
            return parent
    return None


def parse_latexml_html(html: str) -> list[dict] | None:
    """Split an arXiv/ar5iv HTML rendering into sections, subsections and figure captions."""
    from bs4 import BeautifulSoup

    if "ltx_document" not in html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for m in soup.find_all("math"):  # MathML -> its LaTeX source, much friendlier for search
        m.replace_with(f" {m.get('alttext', '')} ")
    for junk in soup.select(".ltx_bibliography, .ltx_note, script, style, nav, .ltx_page_footer"):
        junk.decompose()

    sections: list[dict] = []
    path_of: dict[int, list[str]] = {}

    def heading(el) -> str:
        h = el.find(re.compile(r"^h[1-6]$"), class_="ltx_title", recursive=False) or el.find(
            re.compile(r"^h[1-6]$"), class_="ltx_title"
        )
        if not h:
            return "Untitled"
        for tag in h.select(".ltx_tag"):
            tag.decompose()
        return re.sub(r"^(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+", "", clean_ws(h.get_text(" ")))

    def walk(el, parent_path: list[str]):
        title = heading(el)
        if SKIP_SECTION.match(title):
            return
        path = parent_path + [title]
        path_of[id(el)] = path
        paras = [
            clean_ws(p.get_text(" "))
            for p in el.find_all("div", class_="ltx_para")
            if _section_ancestor(p) is el and p.find_parent("figure") is None
        ]
        paras = [p for p in paras if p]
        if paras:
            sections.append({"path": path, "kind": "text", "paragraphs": paras})
        for child in el.find_all("section"):
            if "ltx_paragraph" not in _classes(child) and _section_ancestor(child) is el:
                walk(child, path)

    tops = [
        s
        for s in soup.find_all("section")
        if ("ltx_section" in _classes(s) or "ltx_appendix" in _classes(s)) and _section_ancestor(s) is None
    ]
    for s in tops:
        walk(s, [])
    if not sections:
        return None

    for fig in soup.find_all("figure"):
        if fig.find_parent("figure") is not None:
            continue  # subfigure; outer caption covers it
        cap = fig.find("figcaption")
        if not cap:
            continue
        sec = _section_ancestor(fig)
        path = path_of.get(id(sec)) if sec is not None else None
        if path is None:
            continue  # figure in a skipped section
        sections.append({"path": path, "kind": "figure", "paragraphs": [clean_ws(cap.get_text(" "))]})
    return sections
