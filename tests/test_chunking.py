"""Tests for rag.chunking.split_documents."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.chunking import split_documents
from rag.pdf_loader import load_pdf_bytes

FILE_HASH = "a" * 64


def _long_page(page: int = 1, sentences: int = 60) -> Document:
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(sentences))
    return Document(
        page_content=text,
        metadata={"source": "long.pdf", "page": page, "file_hash": FILE_HASH, "total_pages": 2},
    )


def test_metadata_preserved_and_ids_added():
    chunks = split_documents([_long_page()], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "long.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["file_hash"] == FILE_HASH
        assert chunk.metadata["total_pages"] == 2
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["chunk_id"] == f"{FILE_HASH[:12]}-p1-c{i}"
        assert len(chunk.page_content) <= 300


def test_does_not_mutate_input_metadata():
    page = _long_page()
    split_documents([page], chunk_size=300, chunk_overlap=50)
    assert "chunk_id" not in page.metadata


def test_consecutive_chunks_overlap():
    chunks = split_documents([_long_page()], chunk_size=300, chunk_overlap=100)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        # the tail of one chunk should reappear at the start of the next
        head = nxt.page_content[:20]
        assert head in prev.page_content, (prev.page_content, nxt.page_content)


def test_zero_overlap_has_no_repeated_text():
    chunks = split_documents([_long_page()], chunk_size=300, chunk_overlap=0)
    joined = " ".join(c.page_content for c in chunks)
    assert joined.count("Sentence number 10 ") == 1


def test_chunk_ids_unique_across_pages(sample_pdf_bytes, second_pdf_bytes):
    pages = []
    for name, data in (sample_pdf_bytes, second_pdf_bytes):
        pages.extend(load_pdf_bytes(name, data).pages)
    pages.extend([_long_page(1), _long_page(2)])
    chunks = split_documents(pages, chunk_size=120, chunk_overlap=20)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    # chunk_index restarts per page
    assert {c.metadata["chunk_index"] for c in chunks if c.metadata["page"] == 1} >= {0, 1}


def test_whitespace_only_pages_produce_no_chunks():
    docs = [Document(page_content="   \n\n  ", metadata={"page": 1, "file_hash": FILE_HASH})]
    assert split_documents(docs, chunk_size=100, chunk_overlap=10) == []


def test_short_page_single_chunk():
    doc = Document(page_content="Short text.", metadata={"page": 4, "file_hash": FILE_HASH})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_id"] == f"{FILE_HASH[:12]}-p4-c0"


@pytest.mark.parametrize("size, overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_chunk_args(size, overlap):
    with pytest.raises(ValueError):
        split_documents([_long_page()], chunk_size=size, chunk_overlap=overlap)
