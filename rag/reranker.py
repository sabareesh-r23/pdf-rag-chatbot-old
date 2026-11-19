"""Cross-encoder reranking of retrieved chunks.

A bi-encoder (the embedding model) scores the question and each chunk separately, so a
chunk whose wording differs from the question, or whose key sentence is a small part of
it, can rank low. A cross-encoder reads the question and the chunk *together* and gives a
much sharper relevance score, but it is too slow to run over the whole index. The usual
pattern is used here: retrieve ~30 candidates cheaply, rerank them, keep the best k.
"""
from __future__ import annotations

from functools import lru_cache


class RerankerInitError(Exception):
    """Raised when the cross-encoder cannot be loaded (message is user-facing)."""


@lru_cache(maxsize=2)
def get_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise RerankerInitError("sentence-transformers is not installed.") from exc
    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RerankerInitError(
            f"Failed to load reranker model '{model_name}': {exc}. The first use downloads it "
            "(~90 MB) from Hugging Face; check the internet connection, or turn reranking off."
        ) from exc


def rerank_scores(model_name: str, question: str, texts: list[str]) -> list[float]:
    """Relevance score of each text for the question (higher = more relevant)."""
    if not texts:
        return []
    model = get_reranker(model_name)
    return [float(s) for s in model.predict([(question, t) for t in texts])]
