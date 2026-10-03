"""BM25 over a precomputed sparse term-weight matrix."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


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
