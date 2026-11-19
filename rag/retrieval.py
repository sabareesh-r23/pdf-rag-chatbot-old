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
    store: "VectorStoreManager",
    question: str,
    k: int,
    file_hashes: list[str] | None = None,
    mode: str = "similarity",
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[RetrievedChunk]:
    """Embed the question and return k chunks, highest cosine similarity first.

    mode="similarity" returns the k most similar chunks; mode="mmr" uses Maximal
    Marginal Relevance over the `fetch_k` most similar chunks to reduce redundancy.
    """
    if question is None or not question.strip():
        raise ValueError("Question must not be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if mode == "similarity":
        results = store.similarity_search(question.strip(), k=k, file_hashes=file_hashes)
    elif mode == "mmr":
        results = store.mmr_search(
            question.strip(), k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, file_hashes=file_hashes
        )
    else:
        raise ValueError(f"Unknown retrieval mode {mode!r}; use 'similarity' or 'mmr'.")
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
