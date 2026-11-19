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
    rerank_score: float | None = None  # cross-encoder relevance, when reranking was used


def retrieve(
    store: "VectorStoreManager",
    question: str,
    k: int,
    file_hashes: list[str] | None = None,
    mode: str = "hybrid",
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    rerank_model: str | None = None,
    rerank_candidates: int = 30,
) -> list[RetrievedChunk]:
    """Return k chunks for the question from the active corpus.

    mode="similarity" returns the k most similar chunks; mode="mmr" uses Maximal
    Marginal Relevance over the `fetch_k` most similar chunks to reduce redundancy;
    mode="hybrid" fuses BM25 keyword and embedding rankings (top `fetch_k` of each).

    With `rerank_model` (similarity / hybrid): `rerank_candidates` chunks are retrieved
    (the hybrid pool grows to that size too), scored by a cross-encoder, and the best k
    are returned in rerank order. With MMR the diverse selection of k is kept and the
    cross-encoder only reorders it; widening MMR to the whole candidate pool would make
    it select every candidate and lose its diversity. Without reranking, chunks are
    returned highest cosine similarity first.
    """
    if question is None or not question.strip():
        raise ValueError("Question must not be empty.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    final_k = k
    if rerank_model and mode != "mmr":
        k = max(k, rerank_candidates)
    if mode == "similarity":
        results = store.similarity_search(question.strip(), k=k, file_hashes=file_hashes)
    elif mode == "mmr":
        results = store.mmr_search(
            question.strip(), k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, file_hashes=file_hashes
        )
    elif mode == "hybrid":
        results = store.hybrid_search(question.strip(), k=k, file_hashes=file_hashes, fetch_k=fetch_k)
    else:
        raise ValueError(f"Unknown retrieval mode {mode!r}; use 'similarity', 'mmr' or 'hybrid'.")
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
    if rerank_model and chunks:
        from rag.reranker import rerank_scores

        for c, r in zip(chunks, rerank_scores(rerank_model, question.strip(), [c.text for c in chunks])):
            c.rerank_score = r
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks[:final_k]
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:final_k]
