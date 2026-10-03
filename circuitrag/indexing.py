"""Build the on-disk index: chunks, tags, embeddings, BM25."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import chunk_context_text, chunk_paper
from .embedding import DEFAULT_MODEL, Embedder
from .tagging import auto_keywords, vocab_tags
from .text import tokenize


def build_index(papers_dir: str | Path, index_dir: str | Path, model: str = DEFAULT_MODEL,
                force_lsa: bool = False, max_words: int = 300, overlap: int = 40,
                k1: float = 1.5, b: float = 0.75) -> None:
    """Chunk every paper, tag it, embed the chunks, and write the index."""
    src, idx = Path(papers_dir), Path(index_dir)
    idx.mkdir(parents=True, exist_ok=True)
    papers = [json.loads(f.read_text()) for f in sorted(src.glob("*.json"))]
    if not papers:
        sys.exit(f"No paper JSON files in {src}. Run `fetch` first.")

    chunks, full_texts = [], []
    for p in papers:
        cs = chunk_paper(p, max_words, overlap)
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
    emb = Embedder.build(model, ctx, idx, force_lsa=force_lsa)
    vecs = emb.encode(ctx, progress=True)
    np.save(idx / "embeddings.npy", vecs)

    print("Building BM25...")
    BM25(k1, b).fit([tokenize(t) for t in ctx]).save(idx)

    with open(idx / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    (idx / "papers.json").write_text(json.dumps(paper_meta, indent=1))
    (idx / "meta.json").write_text(json.dumps({"embedder": emb.meta(), "dim": int(vecs.shape[1]),
                                               "max_words": max_words, "overlap": overlap}, indent=1))
    print(f"Index written to {idx}  (embedder: {emb.backend} / {emb.model_name}, dim {vecs.shape[1]})")
