"""Central application configuration.

All tunable settings are read from environment variables (optionally via a
`.env` file) so that chunking / retrieval / model choices can be changed and
reported during evaluation without code edits.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FALLBACK_ANSWER = (
    "I could not find enough information in the uploaded documents to answer this question."
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # Chunking
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))

    # Embeddings (local Sentence Transformers model)
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Vector store
    chroma_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
    )
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "pdf_corpus"))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    # Optional relevance guard: drop retrieved chunks with cosine similarity below this (0 = disabled)
    min_relevance: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE", 0.0))

    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2:3b"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # OPENAI_API_KEY / OPENAI_BASE_URL are read directly by langchain-openai; never stored here.

    # Ingestion limits
    max_file_mb: int = field(default_factory=lambda: _env_int("MAX_FILE_MB", 50))

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be positive")
        if self.llm_provider not in {"ollama", "openai"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'")

    def as_report_dict(self) -> dict:
        """Configuration snapshot for evaluation reports (contains no secrets)."""
        return asdict(self)


def get_settings(**overrides) -> Settings:
    """Build settings from the environment, optionally overriding fields (used by UI/tests)."""
    base = Settings()
    if not overrides:
        return base
    values = asdict(base)
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
