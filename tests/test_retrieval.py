"""Tests for rag.retrieval.retrieve, using fixture PDFs ingested end to end."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.ingest import ingest_file
from rag.retrieval import RetrievedChunk, retrieve
from rag.vectorstore import VectorStoreManager
from tests.conftest import KNOWN_FACTS


@pytest.fixture
def indexed_store(test_settings, fake_embeddings, sample_pdf_bytes, second_pdf_bytes):
    store = VectorStoreManager(test_settings, embeddings=fake_embeddings)
    r1 = ingest_file(store, *sample_pdf_bytes, test_settings)
    r2 = ingest_file(store, *second_pdf_bytes, test_settings)
    assert r1.status == "indexed" and r2.status == "indexed"
    store.test_hashes = {"sample": r1.file_hash, "second": r2.file_hash}
    store.test_names = {"sample": sample_pdf_bytes[0], "second": second_pdf_bytes[0]}
    return store


@pytest.mark.parametrize("question", list(KNOWN_FACTS))
def test_known_facts_retrieved_on_right_page(indexed_store, test_settings, question):
    expected_text, expected_page = KNOWN_FACTS[question]
    chunks = retrieve(indexed_store, question, k=test_settings.top_k)
    assert chunks, "retrieval returned nothing"
    assert all(isinstance(c, RetrievedChunk) for c in chunks)
    hits = [c for c in chunks if expected_text.lower() in c.text.lower()]
    assert hits, f"expected {expected_text!r} in top-{test_settings.top_k}"
    assert any(c.page == expected_page for c in hits)


def test_results_sorted_by_score(indexed_store):
    chunks = retrieve(indexed_store, next(iter(KNOWN_FACTS)), k=5)
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(c.source in set(indexed_store.test_names.values()) and c.page >= 1 and c.chunk_id for c in chunks)


def test_filter_restricts_to_selected_file(indexed_store):
    # A question about the sample PDF, restricted to the second PDF, returns only second-PDF chunks.
    second_hash = indexed_store.test_hashes["second"]
    question = next(iter(KNOWN_FACTS))
    chunks = retrieve(indexed_store, question, k=10, file_hashes=[second_hash])
    assert chunks and all(c.source == indexed_store.test_names["second"] for c in chunks)
    assert retrieve(indexed_store, question, k=10, file_hashes=[]) == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_raises(indexed_store, question):
    with pytest.raises(ValueError):
        retrieve(indexed_store, question, k=3)


def test_empty_store_returns_nothing(test_settings, fake_embeddings):
    store = VectorStoreManager(test_settings, embeddings=fake_embeddings)
    assert retrieve(store, "anything at all?", k=3) == []


@pytest.mark.slow
def test_real_model_semantic_retrieval(tmp_path):
    """Paraphrased query with little keyword overlap must still find the right chunk."""
    from rag.config import get_settings
    from rag.embeddings import EmbeddingInitError, get_embeddings

    settings = get_settings(chroma_dir=str(tmp_path / "chroma"), collection_name="semantic_test")
    try:
        embeddings = get_embeddings(settings.embedding_model)
    except EmbeddingInitError as exc:
        pytest.skip(f"real embedding model unavailable: {exc}")

    texts = [
        "The automobile's engine overheated because the radiator was leaking coolant.",
        "Our quarterly revenue grew by twelve percent thanks to strong overseas sales.",
        "The physician prescribed antibiotics to treat the bacterial infection.",
        "Heavy rainfall caused the river to burst its banks and flood the village.",
    ]
    docs = [
        Document(
            page_content=t,
            metadata={"source": "s.pdf", "page": i + 1, "file_hash": "f" * 64,
                      "chunk_index": 0, "chunk_id": f"ffffffffffff-p{i + 1}-c0"},
        )
        for i, t in enumerate(texts)
    ]
    store = VectorStoreManager(settings, embeddings=embeddings)
    store.add_chunks(docs)

    cases = {
        "Why did my car get too hot?": 1,
        "How much did the company's income increase?": 2,
        "What medicine did the doctor give for the illness?": 3,
        "What led to the town being under water?": 4,
    }
    for mode in ("similarity", "hybrid"):
        for question, page in cases.items():
            top = retrieve(store, question, k=1, mode=mode)[0]
            assert top.page == page, f"{mode}: {question!r} -> page {top.page}"
            assert 0.0 < top.score <= 1.0001


# --- cross-encoder reranking (V1.3) ---------------------------------------------------

def test_retrieve_with_rerank_reorders_and_truncates(monkeypatch, indexed_store):
    import rag.reranker as rr

    calls = {}

    def fake_scores(model, question, texts):
        calls["n"] = len(texts)
        return [float(i) for i in range(len(texts))]  # last candidate is the most relevant

    monkeypatch.setattr(rr, "rerank_scores", fake_scores)
    plain = retrieve(indexed_store, "photosynthesis", k=10)
    chunks = retrieve(indexed_store, "photosynthesis", k=2, rerank_model="fake", rerank_candidates=10)
    assert calls["n"] == min(10, len(plain))  # reranked the candidate pool, not just k
    assert len(chunks) == 2
    assert chunks[0].rerank_score >= chunks[1].rerank_score
    assert chunks[0].chunk_id == plain[calls["n"] - 1].chunk_id  # best rerank score first


def test_retrieve_without_rerank_has_no_rerank_score(indexed_store):
    assert all(c.rerank_score is None for c in retrieve(indexed_store, "photosynthesis", k=3))


def test_mmr_with_rerank_keeps_mmr_selection(monkeypatch, indexed_store):
    """Reranking must not widen MMR to the candidate pool (that would disable MMR)."""
    import rag.reranker as rr

    seen = {}
    monkeypatch.setattr(rr, "rerank_scores", lambda m, q, texts: seen.setdefault("n", len(texts)) and [0.0] * len(texts) or [0.0] * len(texts))
    mmr = retrieve(indexed_store, "photosynthesis", k=2, mode="mmr", fetch_k=6)
    both = retrieve(indexed_store, "photosynthesis", k=2, mode="mmr", fetch_k=6, rerank_model="fake", rerank_candidates=30)
    assert seen["n"] == 2  # only the MMR selection is reranked
    assert {c.chunk_id for c in both} == {c.chunk_id for c in mmr}
