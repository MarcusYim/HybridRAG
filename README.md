# circuitrag

A small hybrid retrieval system for biological-circuit papers from arXiv: section-aware chunking,
domain tagging, dense + BM25 search with a tunable balance, and Claude-generated answers.

## Install

```bash
pip install -e .            # core only
pip install -e '.[all]'     # + sentence-transformers, pymupdf, anthropic
```

Installing puts a `circuitrag` command on your path; without installing, use
`python -m circuitrag` from the project root. Both run the same entry point.

Only `requests numpy scipy scikit-learn beautifulsoup4` are strictly required. Without
`sentence-transformers` the indexer falls back to a TF-IDF + SVD embedder, which works offline but
is much weaker at paraphrase matching. `pymupdf` is only used when arXiv has no HTML rendering of a
paper. `anthropic` is only needed for `ask`.

## Use

```bash
circuitrag fetch --max 60              # download + split into sections (~3 min)
circuitrag index                       # chunk, tag, embed, build BM25
circuitrag shell                       # interactive search
circuitrag search 'tuning oscillator period' --alpha 0.6
export ANTHROPIC_API_KEY=sk-...
circuitrag ask 'What degradation tags speed up oscillator dynamics?' -k 12
```

Or as a library:

```python
from circuitrag import Searcher, build_index, fetch_corpus

fetch_corpus("data/papers", max_results=60)
build_index("data/papers", "data/index")

s = Searcher("data/index")
for hit in s.search('"integral feedback" organism:yeast', k=5, alpha=0.3):
    print(hit.paper["title"], "|", hit.chunk["path"], hit.score)
```

## Query syntax

| Syntax | Meaning |
|---|---|
| `"exact phrase"` | hard requirement; chunk must contain it verbatim |
| `-word`, `-"phrase"` | exclude |
| `organism:yeast` | tag filter (`organism function mechanism method part kw`) |
| `tag:oscillat` | any tag category |
| `year:>=2020`, `year:2018-2022` | publication year |
| `cat:q-bio.MN`, `author:elowitz` | arXiv metadata |
| `section:methods`, `kind:figure` | where in the paper (`kind`: `abstract`, `text`, `figure`) |
| `-organism:mammalian` | negate any filter |

Quoted phrases and filters are applied as a hard pre-filter, so they override the ranking entirely;
remaining words drive the scoring. `--phrase-scope paper` relaxes quotes to "somewhere in this
paper" instead of "in this chunk".

## Tuning the balance

`--alpha` weights dense (semantic) against BM25 (keyword): `1.0` is pure semantic, `0.0` pure
keyword, `0.5` the default. Use low alpha for part names and jargon (`pBAD`, `ssrA`, `Bxb1`), high
alpha for conceptual questions. `--fusion rrf` switches from weighted score blending to reciprocal
rank fusion, which ignores score magnitudes and is more robust when the two scorers are on very
different scales. `--per-paper 2` stops one verbose paper from filling the results.

In `shell`, `:alpha`, `:fusion`, `:k`, `:pp`, `:scope` change these live, and `:ask <question>`
runs generation with the current settings.

## Module layout

| Module | Contents |
|---|---|
| `text.py` | normalization, the section blacklist, the search tokenizer |
| `parse_html.py` | LaTeXML/ar5iv section tree (preferred full-text source) |
| `parse_pdf.py` | heuristic section splitting (fallback) |
| `fetch.py` | arXiv API search, full-text download, `fetch_corpus` |
| `chunking.py` | section-aware packing into embeddable chunks |
| `tagging.py` | regex vocabulary + TF-IDF keyphrases |
| `embedding.py` | `Embedder`: sentence-transformers, or TF-IDF+SVD offline |
| `bm25.py` | `BM25` over a precomputed sparse matrix |
| `indexing.py` | `build_index`: ties the above together, writes to disk |
| `query.py` | `ParsedQuery`: quotes, negation, field filters |
| `search.py` | `Searcher`: dense + sparse fusion over a built index |
| `display.py` | terminal rendering |
| `generate.py` | grounded answer generation |
| `cli.py` | argparse front end |

Dependencies run one way: `text` is the leaf, `cli` is the root, and nothing
imports upward. The heavy optional imports (`sentence_transformers`, `fitz`,
`anthropic`) stay inside the functions that need them, so the package loads
fast and the core works without them.

## How it works

**Chunking** follows the paper's own structure. The arXiv HTML (LaTeXML) rendering is parsed into a
section tree, MathML is replaced by its LaTeX source, and references/acknowledgements are dropped.
Paragraphs are packed into ~300-word chunks with ~40-word overlap, never crossing a subsection
boundary. Figure captions become separate chunks tagged to their enclosing section. Every chunk is
embedded together with its paper title and section path, so a chunk about "the repressor" still
carries its context.

**Tagging** combines a curated regex vocabulary (organism, circuit function, mechanism, method,
parts) with TF-IDF keyphrases extracted per paper relative to the corpus. A vocabulary tag applies
if the term appears in the title or abstract, or at least twice in the body. Part patterns are
case-sensitive so `cI` is not confused with `CI`.

**Retrieval** scores each chunk with cosine similarity against the dense index and with BM25 over a
token stream that splits hyphenated terms (so `feed-forward`, `feedforward` and `feed forward` all
match). Both are min-max normalized before blending.

## Known limits

- The default arXiv query reaches the quantitative-biology and control-theory side of the field.
  Most wet-lab circuit work is in journals, not on arXiv — point `fetch --query` elsewhere or add a
  PubMed/bioRxiv loader for broader coverage.
- Heading detection in the PDF fallback is heuristic; the HTML path is much more reliable.
- There's no reranker. Adding a cross-encoder over the top ~50 hits before generation is the single
  biggest quality win available.
- Tables and DNA sequences are treated as prose, which is wrong for both.
