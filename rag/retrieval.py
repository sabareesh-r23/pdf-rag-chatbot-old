"""Semantic retrieval over the active corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.vectorstore import VectorStoreManager


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float
    chunk_id: str


def retrieve(
    store: "VectorStoreManager", question: str, k: int, file_hashes: list[str] | None = None
) -> list[RetrievedChunk]:
    """Embed the question and return the top-k chunks, highest cosine similarity first."""
    if question is None or not question.strip():
        raise ValueError("Question must not be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    results = store.similarity_search(question.strip(), k=k, file_hashes=file_hashes)
    chunks = [
        RetrievedChunk(
            text=doc.page_content,
            source=str(doc.metadata.get("source", "unknown")),
            page=int(doc.metadata.get("page", 0)),
            score=float(score),
            chunk_id=str(doc.metadata.get("chunk_id", "")),
        )
        for doc, score in results
    ]
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks
