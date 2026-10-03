"""Google-style query parsing.

Quoted phrases and field filters become a hard pre-filter rather than a score
boost, so an override actually overrides.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import clean_ws


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


def year_matches(year: int | None, spec: str) -> bool:
    if year is None:
        return False
    if m := re.match(r"^(\d{4})-(\d{4})$", spec):
        return int(m.group(1)) <= year <= int(m.group(2))
    if m := re.match(r"^(>=|<=|>|<|=)?(\d{4})$", spec):
        op, y = m.group(1) or "=", int(m.group(2))
        return {"=": year == y, ">": year > y, "<": year < y, ">=": year >= y, "<=": year <= y}[op]
    return True
