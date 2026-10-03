"""Answer generation grounded in retrieved excerpts."""
from __future__ import annotations

import os

from .search import Hit


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
