"""Smoke tests for the Streamlit UI (app.py) using streamlit.testing.v1.AppTest.

Offline: embeddings are replaced by the deterministic HashingEmbeddings from
conftest, the Ollama health check is stubbed, and the LLM is a fake chat model.
"""
from __future__ import annotations

from pathlib import Path

import pytest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

pytest.importorskip("streamlit.testing.v1")
for _mod in ("rag.embeddings", "rag.vectorstore", "rag.ingest", "rag.generation", "rag.pipeline"):
    pytest.importorskip(_mod)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import HashingEmbeddings  # noqa: E402


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    import rag.embeddings
    import rag.generation
    import rag.vectorstore

    chroma_dir = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_DIR", str(chroma_dir))
    monkeypatch.setenv("COLLECTION_NAME", "app_smoke")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    fake = HashingEmbeddings()
    monkeypatch.setattr(rag.embeddings, "get_embeddings", lambda name: fake)
    monkeypatch.setattr(rag.vectorstore, "get_embeddings", lambda name: fake, raising=False)
    monkeypatch.setattr(rag.generation, "check_ollama", lambda settings, timeout=3.0: (True, "stubbed"))

    st.cache_resource.clear()
    st.cache_data.clear()
    yield {"chroma_dir": str(chroma_dir), "embeddings": fake}
    st.cache_resource.clear()
    st.cache_data.clear()


def _run() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_initial_render_empty_corpus(app_env):
    at = _run()
    assert not at.exception, at.exception
    assert at.title[0].value == "PDF RAG Chatbot"
    assert any("Upload and index" in i.value for i in at.info)
    assert at.chat_input[0].disabled  # nothing to chat with yet


def test_render_with_indexed_file_and_ask(app_env, sample_pdf_bytes, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    import rag.generation
    import rag.pipeline
    from rag.config import get_settings
    from rag.ingest import ingest_file
    from rag.vectorstore import VectorStoreManager

    store = VectorStoreManager(get_settings(), embeddings=app_env["embeddings"])
    name, data = sample_pdf_bytes
    report = ingest_file(store, name, data, get_settings())
    assert report.status == "indexed", report.message

    fake_llm = FakeListChatModel(responses=["Stub answer [1]."])
    monkeypatch.setattr(rag.generation, "get_llm", lambda settings: fake_llm)
    monkeypatch.setattr(rag.pipeline, "get_llm", lambda settings: fake_llm, raising=False)

    at = _run()
    assert not at.exception, at.exception
    labels = [cb.label for cb in at.sidebar.checkbox]
    assert any(name in lbl for lbl in labels), labels
    assert not at.chat_input[0].disabled

    question = next(iter(__import__("tests.conftest", fromlist=["KNOWN_FACTS"]).KNOWN_FACTS))
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    msgs = at.session_state["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["chunks"], "expected retrieved evidence"
    assert all(c["source"] == name for c in msgs[-1]["chunks"])


def test_deselect_remove_and_clear(app_env, sample_pdf_bytes, second_pdf_bytes):
    from rag.config import get_settings
    from rag.ingest import ingest_file
    from rag.vectorstore import VectorStoreManager

    store = VectorStoreManager(get_settings(), embeddings=app_env["embeddings"])
    for name, data in (sample_pdf_bytes, second_pdf_bytes):
        assert ingest_file(store, name, data, get_settings()).status == "indexed"

    at = _run()
    assert not at.exception, at.exception
    # Untick every file -> chat disabled with a warning.
    for cb in at.sidebar.checkbox:
        if cb.key and cb.key.startswith("sel_"):
            cb.uncheck()
    at.run()
    assert not at.exception, at.exception
    assert at.chat_input[0].disabled
    assert any("No files are selected" in w.value for w in at.warning)

    # Remove one file via its per-file button.
    rm = next(b for b in at.sidebar.button if b.key and b.key.startswith("rm_"))
    rm.click().run()
    assert not at.exception, at.exception
    assert len(store.list_files()) == 1

    # Clear requires the confirm checkbox.
    at.sidebar.checkbox(key="confirm_clear").check().run()
    next(b for b in at.sidebar.button if b.label == "Clear corpus").click().run()
    assert not at.exception, at.exception
    # reset() recreates the collection, so open a fresh manager to inspect it.
    fresh = VectorStoreManager(get_settings(), embeddings=app_env["embeddings"])
    assert fresh.count() == 0
    assert at.session_state["messages"] == []
    assert at.chat_input[0].disabled
