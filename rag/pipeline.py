"""End-to-end RAG pipeline: retrieve -> (relevance filter) -> generate -> package response."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag.config import FALLBACK_ANSWER, Settings
from rag.generation import generate_answer, get_llm
from rag.retrieval import RetrievedChunk, retrieve


@dataclass
class RAGResponse:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    sources: list[dict]  # unique [{"source", "page"}] in rank order; [] when abstained
    abstained: bool  # True when the answer is (or starts with) the fallback
    config: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    """Lowercase, unify apostrophes, strip punctuation and collapse whitespace."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_FALLBACK_NORM = _normalize(FALLBACK_ANSWER)
# Tolerate small paraphrases of the fallback that small models sometimes produce,
# e.g. "I couldn't find enough information in the documents ...".
_PARAPHRASE_RE = re.compile(
    r"^(sorry |i'm sorry |unfortunately )?i (could not|couldn't|cannot|can't|did not|didn't) find "
    r"(enough|sufficient|any) (relevant )?information"
)


def is_abstention(answer: str) -> bool:
    """True if the answer is the fallback (or a close paraphrase of it) at the start.

    Only the *start* of the answer is checked: a partial answer that first states
    supported facts and then notes a missing part is NOT an abstention.
    """
    norm = _normalize(answer)
    if not norm:
        return True
    return norm.startswith(_FALLBACK_NORM) or bool(_PARAPHRASE_RE.match(norm))


def unique_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    """Unique (source, page) pairs, preserving retrieval rank order."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for c in chunks:
        key = (c.source, c.page)
        if key not in seen:
            seen.add(key)
            out.append({"source": c.source, "page": c.page})
    return out


class RAGPipeline:
    def __init__(self, settings: Settings, store: Any, llm=None):
        self.settings = settings
        self.store = store
        self._llm = llm  # created lazily on first ask() so the UI can start without an LLM

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self.settings)
        return self._llm

    def ask(
        self,
        question: str,
        file_hashes: list[str] | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        if question is None or not str(question).strip():
            raise ValueError("Please enter a question.")
        question = str(question).strip()
        k = top_k if top_k is not None else self.settings.top_k
        if k <= 0:
            raise ValueError("top_k must be positive")

        chunks = retrieve(self.store, question, k, file_hashes)

        # Optional relevance guard (MIN_RELEVANCE; 0 disables it).
        threshold = self.settings.min_relevance
        if threshold > 0:
            chunks = [c for c in chunks if c.score >= threshold]

        if chunks:
            answer = generate_answer(self.llm, question, chunks)
        else:
            answer = FALLBACK_ANSWER  # nothing relevant: never call the LLM

        abstained = is_abstention(answer)
        config = self.settings.as_report_dict()
        config["top_k"] = k  # effective value for this query
        return RAGResponse(
            question=question,
            answer=answer,
            chunks=chunks,
            sources=[] if abstained else unique_sources(chunks),
            abstained=abstained,
            config=config,
        )
