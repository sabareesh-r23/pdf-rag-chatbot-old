"""Tests for rag.pipeline (retrieval stubbed; fake chat model)."""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag import pipeline as pipeline_mod
from rag.config import FALLBACK_ANSWER, get_settings
from rag.pipeline import RAGPipeline, is_abstention, unique_sources
from rag.retrieval import RetrievedChunk


def chunk(source, page, score=0.9, text="Some text.") -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, page=page, score=score, chunk_id=f"{source}-{page}-{score}")


class CountingLLM(FakeListChatModel):
    n_calls: int = 0

    def invoke(self, *args, **kwargs):
        self.n_calls += 1
        return super().invoke(*args, **kwargs)


@pytest.fixture
def stub_retrieve(monkeypatch):
    """Replace rag.pipeline.retrieve; returns a dict to configure results / inspect calls."""
    state = {"chunks": [], "calls": []}

    def fake(store, question, k, file_hashes=None):
        state["calls"].append({"question": question, "k": k, "file_hashes": file_hashes})
        return list(state["chunks"])[:k]

    monkeypatch.setattr(pipeline_mod, "retrieve", fake)
    return state


# --- helpers ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "answer",
    [
        FALLBACK_ANSWER,
        "  " + FALLBACK_ANSWER.upper() + "  ",
        "i could not find   enough information in the uploaded documents to answer this question",
        "I couldn't find enough information in the provided documents.",
        "Sorry, I could not find any relevant information about that.",
        "",
    ],
)
def test_is_abstention_true(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The reactor runs at 451 degrees [1].",
        # partial answer: supported fact first, then the missing part -> not an abstention
        "Mira Vance founded it [1]. I could not find enough information in the uploaded documents about the budget.",
    ],
)
def test_is_abstention_false(answer):
    assert not is_abstention(answer)


def test_unique_sources_dedupes_in_rank_order():
    chunks = [chunk("b.pdf", 2), chunk("a.pdf", 1), chunk("b.pdf", 2), chunk("a.pdf", 3), chunk("a.pdf", 1)]
    assert unique_sources(chunks) == [
        {"source": "b.pdf", "page": 2},
        {"source": "a.pdf", "page": 1},
        {"source": "a.pdf", "page": 3},
    ]


# --- RAGPipeline.ask ----------------------------------------------------------------

def test_blank_question_raises(stub_retrieve):
    p = RAGPipeline(get_settings(), store=object(), llm=CountingLLM(responses=["x"]))
    for q in ["", "   ", "\n\t"]:
        with pytest.raises(ValueError):
            p.ask(q)
    assert stub_retrieve["calls"] == []


def test_answer_with_sources_and_config(stub_retrieve):
    stub_retrieve["chunks"] = [chunk("b.pdf", 2), chunk("a.pdf", 1), chunk("b.pdf", 2)]
    llm = CountingLLM(responses=["The value is 7 [1]."])
    p = RAGPipeline(get_settings(top_k=4), store="STORE", llm=llm)
    r = p.ask("  What is the value?  ", file_hashes=["h1"], top_k=3)
    assert r.answer == "The value is 7 [1]."
    assert r.question == "What is the value?"
    assert not r.abstained
    assert r.sources == [{"source": "b.pdf", "page": 2}, {"source": "a.pdf", "page": 1}]
    assert len(r.chunks) == 3
    assert r.config["top_k"] == 3 and r.config["llm_model"]
    assert stub_retrieve["calls"] == [{"question": "What is the value?", "k": 3, "file_hashes": ["h1"]}]
    assert llm.n_calls == 1


def test_default_top_k_from_settings(stub_retrieve):
    p = RAGPipeline(get_settings(top_k=6), store=None, llm=CountingLLM(responses=["x"]))
    r = p.ask("q?")
    assert stub_retrieve["calls"][0]["k"] == 6 and r.config["top_k"] == 6


def test_no_chunks_abstains_without_llm(stub_retrieve):
    llm = CountingLLM(responses=["should not be used"])
    r = RAGPipeline(get_settings(), store=None, llm=llm).ask("Unknown?")
    assert r.answer == FALLBACK_ANSWER and r.abstained and r.sources == []
    assert llm.n_calls == 0


def test_model_abstention_clears_sources(stub_retrieve):
    stub_retrieve["chunks"] = [chunk("a.pdf", 1)]
    llm = CountingLLM(responses=[FALLBACK_ANSWER])
    r = RAGPipeline(get_settings(), store=None, llm=llm).ask("Off-topic?")
    assert r.abstained and r.sources == []
    assert len(r.chunks) == 1  # retrieved evidence still exposed for transparency


def test_min_relevance_filters_chunks(stub_retrieve):
    stub_retrieve["chunks"] = [chunk("a.pdf", 1, 0.8), chunk("b.pdf", 2, 0.2), chunk("c.pdf", 3, 0.5)]
    llm = CountingLLM(responses=["Answer [1]."])
    r = RAGPipeline(get_settings(min_relevance=0.5), store=None, llm=llm).ask("q?")
    assert [c.source for c in r.chunks] == ["a.pdf", "c.pdf"]
    assert r.sources == [{"source": "a.pdf", "page": 1}, {"source": "c.pdf", "page": 3}]


def test_min_relevance_dropping_everything_falls_back(stub_retrieve):
    stub_retrieve["chunks"] = [chunk("a.pdf", 1, 0.1)]
    llm = CountingLLM(responses=["nope"])
    r = RAGPipeline(get_settings(min_relevance=0.5), store=None, llm=llm).ask("q?")
    assert r.abstained and r.answer == FALLBACK_ANSWER and llm.n_calls == 0


def test_llm_created_lazily(stub_retrieve, monkeypatch):
    created = []

    def fake_get_llm(settings):
        created.append(settings)
        return CountingLLM(responses=["Lazy [1]."])

    monkeypatch.setattr(pipeline_mod, "get_llm", fake_get_llm)
    p = RAGPipeline(get_settings(), store=None)
    assert created == []
    stub_retrieve["chunks"] = [chunk("a.pdf", 1)]
    p.ask("q?")
    p.ask("q again?")
    assert len(created) == 1


# --- end-to-end with the real (offline) vector store ---------------------------------

def test_end_to_end_with_real_store(test_settings, fake_embeddings, sample_pdf_bytes):
    from rag.ingest import ingest_file
    from rag.vectorstore import VectorStoreManager

    store = VectorStoreManager(test_settings, embeddings=fake_embeddings)
    name, data = sample_pdf_bytes
    report = ingest_file(store, name, data, test_settings)
    assert report.status == "indexed", report.message

    llm = CountingLLM(responses=["Captain Mira Vance founded it [1]."])
    r = RAGPipeline(test_settings, store, llm=llm).ask("Who founded the Lumen Institute?", top_k=2)
    assert not r.abstained
    assert r.chunks and r.chunks[0].source == name
    assert {"source": name, "page": 2} in r.sources
