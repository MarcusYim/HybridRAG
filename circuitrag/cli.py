"""
circuitrag command line.

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
from collections import Counter
import json
from pathlib import Path

from .display import _c, print_hits
from .embedding import DEFAULT_MODEL
from .fetch import DEFAULT_QUERY, fetch_corpus
from .generate import DEFAULT_LLM, answer
from .indexing import build_index
from .query import parse_query
from .search import Hit, Searcher
from .tagging import TAG_VOCAB


def cmd_fetch(args):
    fetch_corpus(args.out, query=args.query, max_results=args.max, sort_by=args.sort,
                 full_text=not args.no_fulltext, refresh=args.refresh, delay=args.delay)


def cmd_index(args):
    build_index(args.papers, args.index, model=args.model, force_lsa=args.lsa,
                max_words=args.max_words, overlap=args.overlap, k1=args.k1, b=args.b)


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
    ap = argparse.ArgumentParser(prog="circuitrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

