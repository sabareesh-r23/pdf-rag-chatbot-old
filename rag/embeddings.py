"""Local Sentence-Transformers embeddings (via LangChain), cached per model name."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings


class EmbeddingInitError(Exception):
    """Raised when the embedding model cannot be imported, downloaded or initialised."""


@lru_cache(maxsize=4)
def get_embeddings(model_name: str) -> Embeddings:
    """Return a (cached) HuggingFaceEmbeddings instance with L2-normalised vectors.

    Normalised vectors make cosine similarity well-defined and comparable across
    queries. A tiny smoke embedding is computed here so that download / load
    failures surface immediately with an actionable message, instead of later
    in the middle of indexing or answering.
    """
    if not model_name or not model_name.strip():
        raise EmbeddingInitError("EMBEDDING_MODEL is empty; set it to a Sentence Transformers model name.")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingInitError(
            "Could not import langchain-huggingface / sentence-transformers. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        embeddings.embed_query("embedding smoke test")
    except Exception as exc:
        raise EmbeddingInitError(
            f"Failed to load embedding model '{model_name}': {exc}. "
            "Check that (1) you have an internet connection for the first download "
            "(the model is cached afterwards), (2) the model name is a valid "
            "Sentence Transformers / Hugging Face model id, and (3) there is enough "
            "free disk space in the Hugging Face cache (~/.cache/huggingface)."
        ) from exc
    return embeddings
