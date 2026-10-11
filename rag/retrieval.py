"""Semantic retrieval over the active corpus."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float
    chunk_id: str


# retrieve() is implemented by the retrieval agent — see CONTRACTS.md
