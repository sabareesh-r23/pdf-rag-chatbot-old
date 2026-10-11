"""Shared pytest fixtures (offline, deterministic).

Stable fixture names other test modules rely on:

* ``sample_pdfs_dir``    - tmp directory where generated PDFs are written.
* ``sample_pdf_bytes``   - ``(filename, bytes)`` for a 3-page text PDF whose facts
                            are listed in the module-level ``KNOWN_FACTS`` dict
                            (question -> (expected substring, 1-based page)).
* ``second_pdf_bytes``   - ``(filename, bytes)`` for a different 2-page PDF
                            (facts in ``SECOND_FACTS``), for multi-file tests.
* ``blank_pdf_bytes``    - ``(filename, bytes)`` for a 2-page PDF with no text.
* ``corrupted_pdf_bytes``- ``(filename, bytes)`` starting with ``%PDF-`` then garbage.
* ``fake_embeddings``    - a ``HashingEmbeddings`` instance (no model download).
* ``test_settings``      - ``Settings`` with chroma_dir in tmp_path and
                            collection_name ``"test_corpus"``.

``HashingEmbeddings`` is a bag-of-words hashing embedding: lowercase word tokens
are hashed into ``dim`` buckets and the vector is L2-normalized, so texts that
share words have high cosine similarity.
"""
from __future__ import annotations

import hashlib
import io
import math
import re

import pytest
from langchain_core.embeddings import Embeddings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag.config import get_settings

# --- Fixture PDF content ----------------------------------------------------

SAMPLE_PAGES: list[str] = [
    # page 1
    "Zorblax Reactor Technical Overview\n"
    "The Zorblax reactor operates at 451 degrees Celsius during normal output.\n"
    "Its cooling loop uses liquid sodium pumped through twelve ceramic channels.\n"
    "Maintenance crews inspect the reactor core every 90 days.",
    # page 2
    "History of the Lumen Institute\n"
    "Captain Mira Vance founded the Lumen Institute in 1987.\n"
    "The institute began as a small observatory on the northern coast.\n"
    "Today it trains astronomers from more than forty countries.",
    # page 3
    "Glacier Bees of Patagonia\n"
    "The glacier bee pollinates only the frost lily, which blooms in March.\n"
    "A single hive of glacier bees contains roughly 3,200 workers.\n"
    "Researchers tag the bees with tiny copper bands to track migration.",
]

KNOWN_FACTS: dict[str, tuple[str, int]] = {
    "At what temperature does the Zorblax reactor operate?": ("451 degrees", 1),
    "Who founded the Lumen Institute?": ("Captain Mira Vance", 2),
    "In what year was the Lumen Institute founded?": ("1987", 2),
    "Which flower does the glacier bee pollinate?": ("frost lily", 3),
}

SECOND_PAGES: list[str] = [
    "Quantum Bakery Operations Manual\n"
    "The Quantum Bakery bakes 640 sourdough loaves every morning.\n"
    "Its signature pastry is the cardamom spiral, invented by chef Odile Brandt.",
    "Delivery Logistics\n"
    "Deliveries leave the bakery at 5:30 AM using six electric cargo bikes.",
]

SECOND_FACTS: dict[str, tuple[str, int]] = {
    "How many sourdough loaves does the Quantum Bakery bake each morning?": ("640", 1),
    "Who invented the cardamom spiral?": ("Odile Brandt", 1),
    "When do deliveries leave the bakery?": ("5:30 AM", 2),
}


def make_pdf(pages: list[str | None]) -> bytes:
    """Render each string as one PDF page (None -> blank page with no text)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        if text:
            y = 720
            for line in text.split("\n"):
                c.drawString(72, y, line)
                y -= 18
        else:
            c.rect(100, 100, 200, 200)  # a drawing, but no extractable text
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def sample_pdfs_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    return d


def _write(dir_, name: str, data: bytes) -> tuple[str, bytes]:
    (dir_ / name).write_bytes(data)
    return name, data


@pytest.fixture
def sample_pdf_bytes(sample_pdfs_dir) -> tuple[str, bytes]:
    return _write(sample_pdfs_dir, "sample_facts.pdf", make_pdf(SAMPLE_PAGES))


@pytest.fixture
def second_pdf_bytes(sample_pdfs_dir) -> tuple[str, bytes]:
    return _write(sample_pdfs_dir, "quantum_bakery.pdf", make_pdf(SECOND_PAGES))


@pytest.fixture
def blank_pdf_bytes(sample_pdfs_dir) -> tuple[str, bytes]:
    return _write(sample_pdfs_dir, "blank_scan.pdf", make_pdf([None, None]))


@pytest.fixture
def corrupted_pdf_bytes(sample_pdfs_dir) -> tuple[str, bytes]:
    data = b"%PDF-1.4\n" + b"\x00garbage\xff\xfe not really a pdf " * 50
    return _write(sample_pdfs_dir, "corrupted.pdf", data)


# --- Offline embeddings ------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (L2-normalized)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0  # avoid zero vectors (undefined cosine)
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings() -> HashingEmbeddings:
    return HashingEmbeddings()


@pytest.fixture
def test_settings(tmp_path):
    return get_settings(chroma_dir=str(tmp_path / "chroma"), collection_name="test_corpus")
