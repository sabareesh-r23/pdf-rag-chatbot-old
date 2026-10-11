"""Tests for rag.ingest.ingest_file status reporting."""
from __future__ import annotations

import pytest

from rag.ingest import IngestReport, ingest_file
from rag.vectorstore import VectorStoreManager
from tests.conftest import SAMPLE_PAGES, make_pdf


@pytest.fixture
def store(test_settings, fake_embeddings):
    return VectorStoreManager(test_settings, embeddings=fake_embeddings)


def test_ingest_indexes_valid_pdf(store, test_settings, sample_pdf_bytes):
    name, data = sample_pdf_bytes
    report = ingest_file(store, name, data, test_settings)
    assert isinstance(report, IngestReport)
    assert report.status == "indexed", report.message
    assert report.filename == name
    assert report.pages == 3
    assert report.empty_pages == []
    assert report.chunks > 0 and report.chunks == store.count()
    assert len(report.file_hash) == 64
    assert store.has_file(report.file_hash)
    assert "Warning" not in report.message


def test_duplicate_upload_is_not_reindexed(store, test_settings, sample_pdf_bytes):
    name, data = sample_pdf_bytes
    first = ingest_file(store, name, data, test_settings)
    count = store.count()
    # Same bytes under a different filename are still a duplicate (hash-based).
    second = ingest_file(store, "renamed.pdf", data, test_settings)
    assert second.status == "duplicate"
    assert second.file_hash == first.file_hash
    assert second.chunks == 0
    assert store.count() == count


def test_multiple_files(store, test_settings, sample_pdf_bytes, second_pdf_bytes):
    a = ingest_file(store, *sample_pdf_bytes, test_settings)
    b = ingest_file(store, *second_pdf_bytes, test_settings)
    assert a.status == b.status == "indexed"
    assert {f["file_hash"] for f in store.list_files()} == {a.file_hash, b.file_hash}


def test_partially_empty_pdf_warns(store, test_settings):
    data = make_pdf([SAMPLE_PAGES[0], None, SAMPLE_PAGES[2]])
    report = ingest_file(store, "partial.pdf", data, test_settings)
    assert report.status == "indexed"
    assert report.pages == 3
    assert report.empty_pages == [2]
    assert "Warning" in report.message


@pytest.mark.parametrize(
    "fixture_name", ["corrupted_pdf_bytes", "blank_pdf_bytes"]
)
def test_bad_pdfs_return_error(store, test_settings, request, fixture_name):
    name, data = request.getfixturevalue(fixture_name)
    report = ingest_file(store, name, data, test_settings)
    assert report.status == "error"
    assert report.message
    assert store.count() == 0


def test_non_pdf_returns_error(store, test_settings):
    report = ingest_file(store, "notes.txt", b"just some text", test_settings)
    assert report.status == "error"
    assert report.message
    assert store.count() == 0


def test_empty_bytes_return_error(store, test_settings):
    report = ingest_file(store, "empty.pdf", b"", test_settings)
    assert report.status == "error"


def test_unexpected_failure_is_reported_not_raised(store, test_settings, sample_pdf_bytes, monkeypatch):
    def boom(chunks):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "add_chunks", boom)
    report = ingest_file(store, *sample_pdf_bytes, test_settings)
    assert report.status == "error"
    assert "disk full" in report.message
