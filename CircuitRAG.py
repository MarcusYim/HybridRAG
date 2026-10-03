#!/usr/bin/env python3
"""
circuitrag: a small hybrid RAG system for biological-circuit papers on arXiv.

Subcommands
  fetch   pull papers from the arXiv API, download full text, split into sections
  index   chunk by section, tag papers, build dense embeddings + BM25 index
  search  hybrid search with a tunable dense/keyword balance and query operators
  ask     retrieve, then have Claude answer from the retrieved excerpts (needs ANTHROPIC_API_KEY)
  shell   interactive search loop for tuning the balance live
  tags    show tag counts across the corpus

Query syntax (Google-style)
  "exact phrase"        must appear verbatim (case-insensitive, word boundaries)
  -word  -"phrase"      must NOT appear
  organism:yeast        filter on a tag category (organism, function, mechanism, method, part)
  tag:oscillat          any tag containing the text;  kw:noise  auto-extracted keyword
  year:>=2020  year:2018-2022  cat:q-bio.MN  author:elowitz  section:methods  kind:figure
  Prefix any filter with - to negate it.  Multi-word values: organism:"E. coli"
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

USER_AGENT = "circuitrag/0.1 (research prototype; mailto:you@example.com)"


def clean_ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def safe_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


SKIP_SECTION = re.compile(
    r"^(references|bibliography|acknowledg|author contributions|competing interests|"
    r"conflicts? of interest|funding|data availability|code availability)",
    re.I,
)

STOPWORDS = set(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each either et al etc
    few for from further had has have having he her here hers him his how however i if in into is it
    its itself just me more most my no nor not now of off on once only or other our ours out over own
    same she should so some such than that the their theirs them then there these they this those
    through thus to too under until up upon us very via was we were what when where which while who
    whom why will with within without would you your fig figure figs eq eqs ref refs table supplementary
    shown show shows using used use uses based one two three also may might well see section paper""".split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; hyphenated terms also emit joined and split forms
    so 'feed-forward' matches 'feedforward' and 'feed forward'."""
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if "-" in t:
            out.append(t)
            out.append(t.replace("-", ""))
            out.extend(p for p in t.split("-") if len(p) > 1 and p not in STOPWORDS)
        elif len(t) > 1 and t not in STOPWORDS:
            out.append(t)
    return out


# ----------------------------------------------------------------------------
# 1. Fetching from arXiv
# ----------------------------------------------------------------------------

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


# --- Full text: arXiv HTML (LaTeXML) first, PDF second ---------------------


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


def cmd_fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Querying arXiv for up to {args.max} papers...")
    papers = arxiv_search(args.query, args.max, sort_by=args.sort)
    print(f"Found {len(papers)} papers.")
    for i, p in enumerate(papers, 1):
        dest = out / f"{safe_name(p['id'])}.json"
        if dest.exists() and not args.refresh:
            print(f"[{i}/{len(papers)}] {p['id']} cached")
            continue
        if args.no_fulltext:
            p["source"], p["sections"] = "abstract", []
        else:
            p["source"], p["sections"] = fetch_fulltext(p)
            time.sleep(args.delay)
        dest.write_text(json.dumps(p, indent=1))
        n = sum(len(s["paragraphs"]) for s in p["sections"])
        print(f"[{i}/{len(papers)}] {p['id']} via {p['source']:8s} {len(p['sections']):3d} sections {n:4d} paras  {p['title'][:60]}")


# ----------------------------------------------------------------------------
# 2. Chunking
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# 3. Tagging: domain vocabulary + auto-extracted keywords
# ----------------------------------------------------------------------------

TAG_VOCAB: dict[str, list[tuple[str, str]]] = {
    "organism": [
        ("E. coli", r"\bE\.\s?coli\b|\bEscherichia coli\b"),
        ("yeast", r"\byeast\b|\bS\.\s?cerevisiae\b|\bSaccharomyces\b"),
        ("mammalian", r"\bmammalian\b|\bHEK\s?-?293\w*|\bCHO cells?\b|\bhuman cells?\b"),
        ("B. subtilis", r"\bB\.\s?subtilis\b|\bBacillus subtilis\b"),
        ("cell-free", r"\bcell[- ]free\b|\bTX-?TL\b|\bPURE system\b"),
        ("plant", r"\bArabidopsis\b|\bNicotiana\b|\bplant cells?\b"),
    ],
    "function": [
        ("toggle switch / bistability", r"\btoggle switch(?:es)?\b|\bbistab\w*"),
        ("oscillator", r"\boscillat\w*|\brepressilator\b"),
        ("logic gate", r"\blogic gates?\b|\b(?:AND|OR|NOR|NAND|XOR|NOT)[- ]gates?\b|\bboolean logic\b"),
        ("feedback control", r"\b(?:negative|positive|integral) feedback\b|\bfeedback control\w*|\bantithetic\b"),
        ("feed-forward loop", r"\bfeed-?forward loops?\b|\bIFFLs?\b"),
        ("memory", r"\b(?:cellular|genetic|molecular|transcriptional) memory\b|\brecording device"),
        ("biosensor", r"\bbiosensors?\b|\bsensing circuits?\b"),
        ("counter", r"\b(?:genetic|molecular) counters?\b"),
        ("pulse / band-pass", r"\bpulse generat\w*|\bedge detect\w*|\bband-?pass\b"),
        ("quorum sensing", r"\bquorum[- ]sensing\b"),
        ("multicellular / consortia", r"\bconsorti(?:a|um)\b|\bco-?cultures?\b|\bmulticellular\b"),
    ],
    "mechanism": [
        ("transcriptional", r"\btranscription(?:al)? (?:factors?|repress\w*|activat\w*|regulat\w*)|\brepressors?\b"),
        ("CRISPR", r"\bCRISPR\w*|\bdCas9\b|\bCas9\b|\bsgRNAs?\b"),
        ("recombinase", r"\brecombinases?\b|\bintegrases?\b|\bBxb1\b"),
        ("RNA-based", r"\briboregulators?\b|\btoehold switch\w*|\bsRNAs?\b|\briboswitch\w*|\bsmall RNAs?\b"),
        ("post-translational", r"\bphosphorylation\b|\bproteases?\b|\bdegrons?\b|\bpost-?translational\b"),
        ("optogenetic", r"\boptogenetic\w*|\blight[- ]induc\w*"),
        ("resource competition / burden", r"\bresource (?:competition|sharing|loading)\b|\bretroactivity\b|\b(?:metabolic|cellular|expression) burden\b"),
    ],
    "method": [
        ("ODE model", r"\bordinary differential equations?\b|\bODEs?\b|\bmass[- ]action\b"),
        ("stochastic model", r"\bstochastic\b|\bGillespie\b|\bchemical master equation\b"),
        ("Hill function", r"\bHill (?:functions?|coefficients?|equations?)\b"),
        ("flow cytometry", r"\bflow cytometry\b|\bFACS\b"),
        ("microfluidics", r"\bmicrofluidic\w*|\bmother machine\b"),
        ("design automation", r"\bCello\b|\bdesign automation\b|\bcomputer-aided design\b"),
        ("parameter inference", r"\bparameter (?:inference|estimation|identification)\b"),
        ("control theory", r"\bcontrol theor\w*|\bPID\b|\brobust (?:control|perfect adaptation)\b"),
    ],
    # Part names are case-sensitive on purpose: cI is not CI, AraC is not "arac".
    "part": [
        (name, pat)
        for name, pat in [
            ("TetR", r"\bTetR\b"), ("LacI", r"\bLacI\b"), ("cI (lambda)", r"\b(?:λ|lambda )?cI\b"),
            ("AraC", r"\bAraC\b"), ("LuxR", r"\bLuxR\b"), ("LuxI", r"\bLuxI\b"), ("PhlF", r"\bPhlF\b"),
            ("SrpR", r"\bSrpR\b"), ("AmtR", r"\bAmtR\b"), ("BetI", r"\bBetI\b"), ("QacR", r"\bQacR\b"),
            ("T7 RNAP", r"\bT7 RNA ?(?:P|polymerase)\b"), ("Ptet", r"\b[pP]\s?[Tt]et[O]?\b"),
            ("Plac", r"\b[pP]\s?lac\b|\bPlacO\b"), ("pBAD", r"\b[pP]BAD\b"), ("Plux", r"\b[pP]\s?lux\b"),
            ("aTc", r"\baTc\b|\banhydrotetracycline\b"), ("IPTG", r"\bIPTG\b"), ("AHL", r"\bAHLs?\b|\bHSL\b"),
            ("arabinose", r"\barabinose\b"), ("GFP", r"\b(?:sf|e)?GFP\b"), ("mCherry", r"\bmCherry\b"),
            ("RFP", r"\bm?RFP\b"), ("YFP", r"\b(?:e)?YFP\b"), ("ssrA / LVA tag", r"\bssrA\b|\bLVA\b|\bLAA\b"),
            ("ECF sigma", r"\bECF sigma\b|\bECF σ"),
        ]
    ],
}
_COMPILED_VOCAB = {
    cat: [(label, re.compile(p, 0 if cat == "part" else re.I)) for label, p in entries]
    for cat, entries in TAG_VOCAB.items()
}


def vocab_tags(paper: dict, full_text: str, min_mentions: int = 2) -> dict[str, list[str]]:
    """Tag if mentioned in title/abstract, or at least `min_mentions` times anywhere."""
    head = f"{paper['title']} {paper['abstract']}"
    tags = {}
    for cat, entries in _COMPILED_VOCAB.items():
        found = []
        for label, rx in entries:
            if rx.search(head) or len(rx.findall(full_text)) >= min_mentions:
                found.append(label)
        tags[cat] = found
    return tags


def auto_keywords(doc_texts: list[str], top_n: int = 8) -> list[list[str]]:
    """Corpus-relative TF-IDF keyphrases per paper (unigrams + bigrams)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(doc_texts)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(STOPWORDS | {"cell", "cells", "gene", "genes", "model", "results", "system", "systems"}),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9-]{2,}\b",
        sublinear_tf=True,
        min_df=2 if n >= 10 else 1,
        max_df=0.5 if n >= 10 else 1.0,
    )
    X = vec.fit_transform(doc_texts)
    terms = np.array(vec.get_feature_names_out())
    out = []
    for i in range(n):
        row = X.getrow(i)
        if row.nnz == 0:
            out.append([])
            continue
        order = np.argsort(-row.data)
        picked: list[str] = []
        for t in terms[row.indices[order]]:
            if any(t in p or p in t for p in picked):  # drop "toggle" if "toggle switch" is in, etc.
                continue
            picked.append(t)
            if len(picked) == top_n:
                break
        out.append(picked)
    return out


# ----------------------------------------------------------------------------
# 4. Dense embeddings (sentence-transformers, with an offline LSA fallback)
# ----------------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    def __init__(self, backend: str, model_name: str, query_prefix: str = "", model=None):
        self.backend, self.model_name, self.query_prefix, self.model = backend, model_name, query_prefix, model

    @staticmethod
    def default_prefix(model_name: str) -> str:
        if "bge" in model_name.lower() and "m3" not in model_name.lower():
            return "Represent this sentence for searching relevant passages: "
        if "e5" in model_name.lower():
            return "query: "
        return ""

    @classmethod
    def build(cls, model_name: str, corpus: list[str], index_dir: Path, force_lsa: bool = False) -> "Embedder":
        if not force_lsa:
            try:
                from sentence_transformers import SentenceTransformer

                return cls("st", model_name, cls.default_prefix(model_name), SentenceTransformer(model_name))
            except Exception as exc:
                print(f"! sentence-transformers unavailable ({type(exc).__name__}: {exc}); using LSA fallback.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=60000, stop_words=list(STOPWORDS))
        X = tfidf.fit_transform(corpus)
        k = max(1, min(256, X.shape[0] - 1, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=k, random_state=0).fit(X)
        with open(index_dir / "lsa.pkl", "wb") as f:
            pickle.dump((tfidf, svd), f)
        return cls("lsa", "tfidf+svd", "", (tfidf, svd))

    @classmethod
    def load(cls, meta: dict, index_dir: Path) -> "Embedder":
        if meta["backend"] == "st":
            from sentence_transformers import SentenceTransformer

            return cls("st", meta["model"], meta["query_prefix"], SentenceTransformer(meta["model"]))
        with open(index_dir / "lsa.pkl", "rb") as f:
            return cls("lsa", meta["model"], "", pickle.load(f))

    def encode(self, texts: list[str], is_query: bool = False, progress: bool = False) -> np.ndarray:
        if self.backend == "st":
            if is_query and self.query_prefix:
                texts = [self.query_prefix + t for t in texts]
            v = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=progress)
            return np.asarray(v, dtype=np.float32)
        tfidf, svd = self.model
        v = svd.transform(tfidf.transform(texts)).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def meta(self) -> dict:
        return {"backend": self.backend, "model": self.model_name, "query_prefix": self.query_prefix}


# ----------------------------------------------------------------------------
# 5. BM25 keyword index (sparse matrix, so scoring is one column-sum)
# ----------------------------------------------------------------------------


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.W = None  # docs x vocab, precomputed BM25 term weights (CSC)

    def fit(self, docs: list[list[str]]) -> "BM25":
        from scipy.sparse import csr_matrix

        rows, cols, tfs, df = [], [], [], Counter()
        lens = np.array([len(d) for d in docs], dtype=np.float32)
        avgdl = float(lens.mean()) if len(lens) else 1.0
        for i, toks in enumerate(docs):
            for t, c in Counter(toks).items():
                j = self.vocab.setdefault(t, len(self.vocab))
                rows.append(i)
                cols.append(j)
                tfs.append(c)
                df[j] += 1
        N = len(docs)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for j, d in df.items():
            idf[j] = np.log(1 + (N - d + 0.5) / (d + 0.5))
        tf = np.array(tfs, dtype=np.float32)
        dl = lens[np.array(rows, dtype=np.int64)] if rows else np.zeros(0, np.float32)
        w = idf[np.array(cols, dtype=np.int64)] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
        self.W = csr_matrix((w, (rows, cols)), shape=(N, len(self.vocab))).tocsc()
        return self

    def score(self, q_tokens: list[str]) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in q_tokens if t in self.vocab})
        if not ids:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()

    def save(self, d: Path):
        from scipy.sparse import save_npz

        save_npz(d / "bm25.npz", self.W)
        (d / "bm25.json").write_text(json.dumps({"k1": self.k1, "b": self.b, "vocab": self.vocab}))

    @classmethod
    def load(cls, d: Path) -> "BM25":
        from scipy.sparse import load_npz

        meta = json.loads((d / "bm25.json").read_text())
        obj = cls(meta["k1"], meta["b"])
        obj.vocab, obj.W = meta["vocab"], load_npz(d / "bm25.npz").tocsc()
        return obj


# ----------------------------------------------------------------------------
# 6. Index build
# ----------------------------------------------------------------------------


def cmd_index(args):
    src, idx = Path(args.papers), Path(args.index)
    idx.mkdir(parents=True, exist_ok=True)
    papers = [json.loads(f.read_text()) for f in sorted(src.glob("*.json"))]
    if not papers:
        sys.exit(f"No paper JSON files in {src}. Run `fetch` first.")

    chunks, full_texts = [], []
    for p in papers:
        cs = chunk_paper(p, args.max_words, args.overlap)
        chunks.extend(cs)
        full_texts.append(" ".join([p["title"], p["abstract"]] + [c["text"] for c in cs]))
    print(f"{len(papers)} papers -> {len(chunks)} chunks "
          f"({Counter(c['kind'] for c in chunks)})")

    print("Tagging...")
    kws = auto_keywords(full_texts)
    paper_meta = {}
    for p, text, kw in zip(papers, full_texts, kws):
        tags = vocab_tags(p, text)
        tags["kw"] = kw
        paper_meta[p["id"]] = {
            k: p.get(k) for k in ("id", "title", "authors", "published", "categories", "url", "source")
        } | {"tags": tags}

    print("Embedding...")
    ctx = [chunk_context_text(c) for c in chunks]
    emb = Embedder.build(args.model, ctx, idx, force_lsa=args.lsa)
    vecs = emb.encode(ctx, progress=True)
    np.save(idx / "embeddings.npy", vecs)

    print("Building BM25...")
    BM25(args.k1, args.b).fit([tokenize(t) for t in ctx]).save(idx)

    with open(idx / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    (idx / "papers.json").write_text(json.dumps(paper_meta, indent=1))
    (idx / "meta.json").write_text(json.dumps({"embedder": emb.meta(), "dim": int(vecs.shape[1]),
                                               "max_words": args.max_words, "overlap": args.overlap}, indent=1))
    print(f"Index written to {idx}  (embedder: {emb.backend} / {emb.model_name}, dim {vecs.shape[1]})")


# ----------------------------------------------------------------------------
# 7. Query parsing and hybrid search
# ----------------------------------------------------------------------------

TAG_FIELDS = {"organism", "function", "mechanism", "method", "part", "kw"}
OTHER_FIELDS = {"tag", "year", "cat", "author", "section", "kind", "paper"}
QUERY_TOKEN = re.compile(r'-?(?:\w+:)?"[^"]*"|\S+')


@dataclass
class ParsedQuery:
    text: str = ""
    phrases: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    filters: list[tuple[str, str, bool]] = field(default_factory=list)  # (field, value, negated)

    def retrieval_text(self) -> str:
        return " ".join([self.text] + self.phrases).strip()


def parse_query(q: str) -> ParsedQuery:
    pq, words = ParsedQuery(), []
    for tok in QUERY_TOKEN.findall(q):
        neg = tok.startswith("-") and len(tok) > 1
        body = tok[1:] if neg else tok
        m = re.match(r"^(\w+):(.+)$", body)
        if m and m.group(1).lower() in TAG_FIELDS | OTHER_FIELDS:
            pq.filters.append((m.group(1).lower(), m.group(2).strip('"'), neg))
        elif len(body) >= 2 and body.startswith('"') and body.endswith('"'):
            phrase = clean_ws(body[1:-1])
            if phrase:
                (pq.exclude if neg else pq.phrases).append(phrase)
        elif neg:
            pq.exclude.append(body.strip('"'))
        else:
            words.append(body.strip('"'))
    pq.text = " ".join(w for w in words if w)
    return pq


def phrase_regex(phrase: str) -> re.Pattern:
    parts = [re.escape(w) for w in phrase.lower().split()]
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\s-]+".join(parts) + r"(?![A-Za-z0-9])")


def _year_ok(year: int | None, spec: str) -> bool:
    if year is None:
        return False
    if m := re.match(r"^(\d{4})-(\d{4})$", spec):
        return int(m.group(1)) <= year <= int(m.group(2))
    if m := re.match(r"^(>=|<=|>|<|=)?(\d{4})$", spec):
        op, y = m.group(1) or "=", int(m.group(2))
        return {"=": year == y, ">": year > y, "<": year < y, ">=": year >= y, "<=": year <= y}[op]
    return True


def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


@dataclass
class Hit:
    chunk: dict
    paper: dict
    score: float
    dense: float
    keyword: float


class Searcher:
    def __init__(self, index_dir: str | Path):
        d = Path(index_dir)
        if not (d / "meta.json").exists():
            sys.exit(f"No index at {d}. Run `index` first.")
        self.meta = json.loads((d / "meta.json").read_text())
        self.chunks = [json.loads(l) for l in open(d / "chunks.jsonl")]
        self.papers = json.loads((d / "papers.json").read_text())
        self.emb = np.load(d / "embeddings.npy")
        self.embedder = Embedder.load(self.meta["embedder"], d)
        self.bm25 = BM25.load(d)
        self.lower = [clean_ws(c["text"]).lower() for c in self.chunks]
        by_paper = defaultdict(list)
        for c, low in zip(self.chunks, self.lower):
            by_paper[c["paper_id"]].append(low)
        self.paper_lower = {pid: c_title + " " + " ".join(v) for pid, v in by_paper.items()
                            for c_title in [self.papers[pid]["title"].lower()]}

    # -- filtering ------------------------------------------------------------
    def _filter_ok(self, c: dict, p: dict, fld: str, val: str) -> bool:
        v = val.lower()
        tags = p["tags"]
        if fld in TAG_FIELDS:
            return any(v in t.lower() for t in tags.get(fld, []))
        if fld == "tag":
            return any(v in t.lower() for vals in tags.values() for t in vals)
        if fld == "year":
            return _year_ok(c["year"], val)
        if fld == "cat":
            return any(cat.lower().startswith(v) for cat in p.get("categories") or [])
        if fld == "author":
            return any(v in a.lower() for a in p.get("authors") or [])
        if fld == "section":
            return v in c["path"].lower()
        if fld == "kind":
            return c["kind"].startswith(v)
        if fld == "paper":
            return c["paper_id"].startswith(val)
        return True

    def mask(self, pq: ParsedQuery, phrase_scope: str = "chunk") -> np.ndarray:
        req = [phrase_regex(s) for s in pq.phrases]
        exc = [phrase_regex(s) for s in pq.exclude]
        out = np.zeros(len(self.chunks), dtype=bool)
        for i, c in enumerate(self.chunks):
            p = self.papers[c["paper_id"]]
            if not all(self._filter_ok(c, p, f, v) != neg for f, v, neg in pq.filters):
                continue
            text = self.paper_lower[c["paper_id"]] if phrase_scope == "paper" else self.lower[i]
            if any(not r.search(text) for r in req) or any(r.search(self.lower[i]) for r in exc):
                continue
            out[i] = True
        return out

    # -- search -----------------------------------------------------------------
    def search(self, query: str, k: int = 10, alpha: float = 0.5, fusion: str = "weighted",
               per_paper: int | None = None, phrase_scope: str = "chunk", rrf_k: int = 60) -> list[Hit]:
        """alpha = weight on dense/semantic score; 1-alpha on BM25 keyword score."""
        pq = parse_query(query)
        cand = np.flatnonzero(self.mask(pq, phrase_scope))
        if cand.size == 0:
            return []
        qtext = pq.retrieval_text()
        if not qtext:  # filters only: newest first, abstracts first
            order = sorted(cand, key=lambda i: (-(self.chunks[i]["year"] or 0), self.chunks[i]["kind"] != "abstract"))
            dense = sparse = score = np.zeros(len(cand))
            ranked = [(int(i), 0.0, 0.0, 0.0) for i in order]
        else:
            dense = self.emb[cand] @ self.embedder.encode([qtext], is_query=True)[0]
            sparse = self.bm25.score(tokenize(qtext))[cand]
            if fusion == "rrf":
                rd = np.empty(len(cand)); rd[np.argsort(-dense)] = np.arange(1, len(cand) + 1)
                rs = np.empty(len(cand)); rs[np.argsort(-sparse)] = np.arange(1, len(cand) + 1)
                score = alpha / (rrf_k + rd) + (1 - alpha) * np.where(sparse > 0, 1 / (rrf_k + rs), 0)
            else:
                score = alpha * minmax(dense) + (1 - alpha) * minmax(sparse)
            order = np.argsort(-score)
            ranked = [(int(cand[j]), float(score[j]), float(dense[j]), float(sparse[j])) for j in order]

        hits, seen = [], Counter()
        for i, s, d, kw in ranked:
            c = self.chunks[i]
            if per_paper and seen[c["paper_id"]] >= per_paper:
                continue
            seen[c["paper_id"]] += 1
            hits.append(Hit(c, self.papers[c["paper_id"]], s, d, kw))
            if len(hits) >= k:
                break
        return hits


# ----------------------------------------------------------------------------
# 8. Display
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# 9. Generation (the "G" in RAG)
# ----------------------------------------------------------------------------

DEFAULT_LLM = os.environ.get("CIRCUITRAG_MODEL", "claude-sonnet-5-5")
SYSTEM_PROMPT = (
    "You are a research assistant for synthetic biology and biological circuit design. Answer the "
    "question using only the numbered paper excerpts provided. Cite excerpts inline as [n]. When "
    "excerpts disagree, say so. If they don't contain enough to answer, say what's missing rather "
    "than filling gaps from memory."
)


def answer(question: str, hits: list[Hit], model: str = DEFAULT_LLM) -> str:
    import anthropic

    sources = "\n\n".join(
        f"[{n}] {h.paper['title']} (arXiv:{h.paper['id']}, {h.chunk['year']}), {h.chunk['kind']}: {h.chunk['path']}\n{h.chunk['text']}"
        for n, h in enumerate(hits, 1)
    )
    msg = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<excerpts>\n{sources}\n</excerpts>\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def run_ask(s: Searcher, question: str, k: int, alpha: float, fusion: str, per_paper: int | None,
            phrase_scope: str, model: str):
    hits = s.search(question, k=k, alpha=alpha, fusion=fusion, per_paper=per_paper, phrase_scope=phrase_scope)
    if not hits:
        print("Nothing retrieved; loosen the filters or quotes.")
        return
    clean_q = parse_query(question).retrieval_text() or question
    print(_c("Retrieved:", "2"))
    for n, h in enumerate(hits, 1):
        print(_c(f"  [{n}] {h.paper['title'][:70]} — {h.chunk['path'][:40]}", "2"))
    print()
    print(answer(clean_q, hits, model))


# ----------------------------------------------------------------------------
# 10. CLI
# ----------------------------------------------------------------------------


def cmd_search(args):
    s = Searcher(args.index)
    print_hits(s.search(args.query, args.k, args.alpha, args.fusion, args.per_paper, args.phrase_scope), args.query)


def cmd_ask(args):
    s = Searcher(args.index)
    run_ask(s, args.question, args.k, args.alpha, args.fusion, args.per_paper, args.phrase_scope, args.model)


def cmd_tags(args):
    papers = json.loads((Path(args.index) / "papers.json").read_text())
    cats = [args.category] if args.category else list(TAG_VOCAB) + ["kw"]
    for cat in cats:
        counts = Counter(t for p in papers.values() for t in p["tags"].get(cat, []))
        print(_c(f"{cat} ({len(counts)})", "1"))
        for t, n in counts.most_common(args.top):
            print(f"  {n:4d}  {t}")


SHELL_HELP = """Type a query to search. Settings:
  :alpha 0.3        dense weight (1 = pure semantic, 0 = pure keyword)
  :fusion rrf       weighted | rrf
  :k 10             number of results
  :pp 2             max results per paper (0 = unlimited)
  :scope paper      where "quoted phrases" must match: chunk | paper
  :ask <question>   retrieve with current settings and answer with Claude
  :show N           full text of result N from the last search
  :tags [category]  tag counts
  :set              show current settings
  :q                quit"""


def cmd_shell(args):
    s = Searcher(args.index)
    st = {"alpha": args.alpha, "fusion": args.fusion, "k": args.k, "pp": args.per_paper or 0, "scope": args.phrase_scope}
    last: list[Hit] = []
    print(f"{len(s.chunks)} chunks from {len(s.papers)} papers. :help for commands.")
    while True:
        try:
            line = input(_c(f"[α={st['alpha']:.2f} {st['fusion']}] ", "32") + "> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, rest = line[1:].partition(" ")
            try:
                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd == "help":
                    print(SHELL_HELP)
                elif cmd == "alpha":
                    st["alpha"] = min(1.0, max(0.0, float(rest)))
                elif cmd == "fusion" and rest in ("weighted", "rrf"):
                    st["fusion"] = rest
                elif cmd == "k":
                    st["k"] = int(rest)
                elif cmd == "pp":
                    st["pp"] = int(rest)
                elif cmd == "scope" and rest in ("chunk", "paper"):
                    st["scope"] = rest
                elif cmd == "set":
                    print(st)
                elif cmd == "show":
                    h = last[int(rest) - 1]
                    print(_c(f"{h.paper['title']} — {h.chunk['path']}", "1") + f"\n{h.paper['url']}\n\n{h.chunk['text']}\n")
                elif cmd == "tags":
                    cmd_tags(argparse.Namespace(index=args.index, category=rest or None, top=15))
                elif cmd == "ask":
                    run_ask(s, rest, st["k"], st["alpha"], st["fusion"], st["pp"] or None, st["scope"], args.model)
                else:
                    print("Unknown command. :help")
            except (ValueError, IndexError) as exc:
                print(f"Bad argument: {exc}")
            continue
        last = s.search(line, st["k"], st["alpha"], st["fusion"], st["pp"] or None, st["scope"])
        print_hits(last, line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download papers from arXiv")
    f.add_argument("--query", default=DEFAULT_QUERY, help="arXiv API search_query string")
    f.add_argument("--max", type=int, default=60)
    f.add_argument("--sort", default="relevance", choices=["relevance", "submittedDate", "lastUpdatedDate"])
    f.add_argument("--out", default="data/papers")
    f.add_argument("--no-fulltext", action="store_true", help="abstracts only (fast)")
    f.add_argument("--refresh", action="store_true", help="re-download cached papers")
    f.add_argument("--delay", type=float, default=1.0, help="seconds between full-text downloads")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("index", help="chunk, tag, embed, and build BM25")
    i.add_argument("--papers", default="data/papers")
    i.add_argument("--index", default="data/index")
    i.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    i.add_argument("--lsa", action="store_true", help="force the offline TF-IDF+SVD embedder")
    i.add_argument("--max-words", type=int, default=300)
    i.add_argument("--overlap", type=int, default=40)
    i.add_argument("--k1", type=float, default=1.5)
    i.add_argument("--b", type=float, default=0.75)
    i.set_defaults(func=cmd_index)

    def search_opts(p):
        p.add_argument("--index", default="data/index")
        p.add_argument("-k", type=int, default=8)
        p.add_argument("--alpha", type=float, default=0.5, help="dense weight: 1 semantic only, 0 keyword only")
        p.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"])
        p.add_argument("--per-paper", type=int, default=None, help="cap results per paper")
        p.add_argument("--phrase-scope", default="chunk", choices=["chunk", "paper"])
        p.add_argument("--model", default=DEFAULT_LLM, help="Claude model for answers")

    s = sub.add_parser("search", help="hybrid search")
    s.add_argument("query")
    search_opts(s)
    s.set_defaults(func=cmd_search)

    a = sub.add_parser("ask", help="retrieve + answer with Claude")
    a.add_argument("question")
    search_opts(a)
    a.set_defaults(func=cmd_ask)

    sh = sub.add_parser("shell", help="interactive search")
    search_opts(sh)
    sh.set_defaults(func=cmd_shell)

    t = sub.add_parser("tags", help="tag counts")
    t.add_argument("--index", default="data/index")
    t.add_argument("--category", choices=list(TAG_VOCAB) + ["kw"])
    t.add_argument("--top", type=int, default=20)
    t.set_defaults(func=cmd_tags)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()