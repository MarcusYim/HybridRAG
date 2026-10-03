"""Hybrid dense + BM25 retrieval over a built index."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embedding import Embedder
from .query import TAG_FIELDS, ParsedQuery, parse_query, phrase_regex, year_matches
from .text import clean_ws, tokenize


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
            return year_matches(c["year"], val)
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
