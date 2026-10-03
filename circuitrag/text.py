"""Shared text normalization, the section blacklist, and the search tokenizer."""
from __future__ import annotations

import re


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
