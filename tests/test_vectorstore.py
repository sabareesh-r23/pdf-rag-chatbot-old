"""Tests for the Chroma-backed VectorStoreManager (fake embeddings, temp dirs)."""
from __future__ import annotations

from langchain_core.documents import Document

from rag.vectorstore import VectorStoreManager


def _chunk(file_hash: str, source: str, page: int, idx: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page": page,
            "file_hash": file_hash,
            "chunk_index": idx,
            "chunk_id": f"{file_hash[:12]}-p{page}-c{idx}",
        },
    )


HASH_A = "a" * 64
HASH_B = "b" * 64

DOC_A = [
    _chunk(HASH_A, "alpha.pdf", 1, 0, "Photosynthesis converts sunlight into chemical energy in plants."),
    _chunk(HASH_A, "alpha.pdf", 2, 0, "Chlorophyll absorbs mostly blue and red light."),
]
DOC_B = [
    _chunk(HASH_B, "beta.pdf", 1, 0, "The French Revolution began in 1789 with the storming of the Bastille."),
    _chunk(HASH_B, "beta.pdf", 1, 1, "Napoleon rose to power after the revolution."),
    _chunk(HASH_B, "beta.pdf", 3, 0, "The Treaty of Versailles was signed in 1919."),
]


def _store(test_settings, fake_embeddings) -> VectorStoreManager:
    return VectorStoreManager(test_settings, embeddings=fake_embeddings)


def test_add_and_count(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    assert store.count() == 0
    assert store.add_chunks(DOC_A) == 2
    assert store.add_chunks(DOC_B) == 3
    assert store.count() == 5
    assert store.add_chunks([]) == 0


def test_readding_same_chunks_is_idempotent(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    store.add_chunks(DOC_A)
    store.add_chunks(DOC_A)
    assert store.count() == 2


def test_persistence_across_instances(test_settings, fake_embeddings):
    _store(test_settings, fake_embeddings).add_chunks(DOC_A)
    reopened = _store(test_settings, fake_embeddings)
    assert reopened.count() == 2
    assert reopened.has_file(HASH_A)


def test_has_file_and_list_files(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    assert not store.has_file(HASH_A)
    assert store.list_files() == []
    store.add_chunks(DOC_A + DOC_B)
    assert store.has_file(HASH_A) and store.has_file(HASH_B)
    assert not store.has_file("c" * 64)
    files = {f["file_hash"]: f for f in store.list_files()}
    assert files[HASH_A] == {"source": "alpha.pdf", "file_hash": HASH_A, "chunks": 2}
    assert files[HASH_B] == {"source": "beta.pdf", "file_hash": HASH_B, "chunks": 3}


def test_reset_clears_everything(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    store.add_chunks(DOC_A + DOC_B)
    store.reset()
    assert store.count() == 0
    assert store.list_files() == []
    assert not store.has_file(HASH_A)
    assert store.similarity_search("photosynthesis", k=4) == []
    # A fresh instance on the same directory must also see an empty corpus.
    assert _store(test_settings, fake_embeddings).count() == 0
    # The store remains usable after reset, and only new data is retrievable.
    store.add_chunks(DOC_B)
    results = store.similarity_search("photosynthesis sunlight", k=10)
    assert results and all(d.metadata["file_hash"] == HASH_B for d, _ in results)


def test_remove_file(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    store.add_chunks(DOC_A + DOC_B)
    assert store.remove_file(HASH_A) == 2
    assert not store.has_file(HASH_A)
    assert store.count() == 3
    assert store.remove_file(HASH_A) == 0


def test_similarity_search_scores_sorted_and_relevant(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    store.add_chunks(DOC_A + DOC_B)
    results = store.similarity_search("When was the Treaty of Versailles signed?", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0001 for s in scores)
    assert "Versailles" in results[0][0].page_content


def test_file_hash_filter_isolates_corpus(test_settings, fake_embeddings):
    store = _store(test_settings, fake_embeddings)
    store.add_chunks(DOC_A + DOC_B)
    # Query about document A's content, but restricted to document B.
    results = store.similarity_search("photosynthesis chlorophyll sunlight", k=10, file_hashes=[HASH_B])
    assert results
    assert all(doc.metadata["file_hash"] == HASH_B for doc, _ in results)
    # Multi-hash filter ($in) returns from both.
    both = store.similarity_search("photosynthesis", k=10, file_hashes=[HASH_A, HASH_B])
    assert {d.metadata["file_hash"] for d, _ in both} == {HASH_A, HASH_B}
    # Empty selection means no documents.
    assert store.similarity_search("photosynthesis", k=10, file_hashes=[]) == []


def test_second_instance_survives_reset_by_another(test_settings, fake_embeddings, sample_pdf_bytes):
    """A reset via one manager must not break another manager on the same collection."""
    from rag.ingest import ingest_file

    a = VectorStoreManager(test_settings, embeddings=fake_embeddings)
    b = VectorStoreManager(test_settings, embeddings=fake_embeddings)
    name, data = sample_pdf_bytes
    ingest_file(a, name, data, test_settings)
    assert b.count() > 0
    a.reset()
    assert b.count() == 0
    assert b.similarity_search("reactor", k=2) == []
