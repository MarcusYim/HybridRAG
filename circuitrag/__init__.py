"""circuitrag: hybrid retrieval over biological-circuit papers from arXiv.

Module layout
  text        normalization, the section blacklist, the search tokenizer
  parse_html  LaTeXML/ar5iv section tree  (preferred full-text source)
  parse_pdf   heuristic section splitting (fallback)
  fetch       arXiv API + full-text download
  chunking    section-aware packing into embeddable chunks
  tagging     regex vocabulary + TF-IDF keyphrases
  embedding   Embedder: sentence-transformers, or TF-IDF+SVD offline
  bm25        BM25 over a precomputed sparse matrix
  indexing    build_index: ties the above together, writes to disk
  query       ParsedQuery: quotes, negation, field filters
  search      Searcher: dense + sparse fusion over a built index
  display     terminal rendering
  generate    grounded answer generation
  cli         argparse front end
"""

from .bm25 import BM25
from .embedding import Embedder
from .fetch import fetch_corpus
from .indexing import build_index
from .query import ParsedQuery, parse_query
from .search import Hit, Searcher

__version__ = "0.1.0"
__all__ = [
    "BM25",
    "Embedder",
    "Hit",
    "ParsedQuery",
    "Searcher",
    "build_index",
    "fetch_corpus",
    "parse_query",
]
