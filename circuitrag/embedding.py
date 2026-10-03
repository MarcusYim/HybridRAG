"""Dense embedding backends.

Prefers sentence-transformers; falls back to TF-IDF + SVD so the tool runs with
no model download. The active backend is recorded in the index so queries can
never be encoded by a different model than the documents were.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .text import STOPWORDS


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
