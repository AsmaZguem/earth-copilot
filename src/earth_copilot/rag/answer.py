"""Retrieve catalogue entries, then let the LLM recommend an index citing them."""

from dataclasses import dataclass

from earth_copilot.llm import get_chat_model
from earth_copilot.rag.retrieve import Hit, Mode, search

SYSTEM_PROMPT = """You help choose spectral indices for Sentinel-2 satellite analysis.
Use only the numbered catalogue entries you are given, and cite them like [1].
Recommend one index, explain in one or two sentences why it fits, and give its formula and bands.
If no entry fits the question, say so instead of guessing."""


@dataclass
class Answer:
    text: str
    sources: list[Hit]


def answer(question: str, k: int = 5, mode: Mode = "hybrid") -> Answer:
    hits = search(question, k=k, mode=mode)
    context = "\n".join(f"[{i}] {h.text} Reference: {h.reference}" for i, h in enumerate(hits, 1))
    reply = get_chat_model().invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Catalogue entries:\n{context}\n\nQuestion: {question}"),
        ]
    )
    return Answer(text=reply.content, sources=hits)
