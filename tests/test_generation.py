"""Tests for rag.generation (no network, no real LLM)."""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag import generation
from rag.config import FALLBACK_ANSWER, get_settings
from rag.generation import (
    SYSTEM_PROMPT,
    LLMInitError,
    check_ollama,
    format_context,
    generate_answer,
    get_llm,
)
from rag.retrieval import RetrievedChunk


class RecordingLLM(FakeListChatModel):
    """Fake chat model that records every message list it receives."""

    calls: list = []

    def invoke(self, input: Any, *args, **kwargs):  # noqa: A002
        self.calls.append(input)
        return super().invoke(input, *args, **kwargs)


def make_llm(*responses: str) -> RecordingLLM:
    return RecordingLLM(responses=list(responses) or ["ok"], calls=[])


def chunk(text="Alpha fact.", source="a.pdf", page=1, score=0.9, cid="x-p1-c0") -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, page=page, score=score, chunk_id=cid)


# --- system prompt / context ----------------------------------------------------

def test_system_prompt_contains_grounding_rules():
    p = SYSTEM_PROMPT.lower()
    assert FALLBACK_ANSWER in SYSTEM_PROMPT
    assert "only" in p and "context" in p
    assert "outside or general knowledge" in p
    assert "[1]" in SYSTEM_PROMPT  # citation format
    assert "part of the question" in p or "cover only some" in p  # partial answers
    assert "not instructions" in p  # prompt-injection hygiene


def test_format_context_numbers_blocks_with_source_and_page():
    ctx = format_context([chunk("First text.", "a.pdf", 3), chunk("Second text.", "b.pdf", 7)])
    assert "[1] (source: a.pdf, page 3)\nFirst text." in ctx
    assert "[2] (source: b.pdf, page 7)\nSecond text." in ctx
    assert ctx.index("[1]") < ctx.index("[2]")


# --- generate_answer -------------------------------------------------------------

def test_no_chunks_returns_fallback_without_calling_llm():
    llm = make_llm("should not be used")
    assert generate_answer(llm, "Anything?", []) == FALLBACK_ANSWER
    assert llm.calls == []


def test_generate_answer_sends_system_and_context_messages_and_strips():
    llm = make_llm("  The answer is 42 [1].  \n")
    out = generate_answer(llm, "What is the answer?", [chunk("The answer is 42.", "doc.pdf", 5)])
    assert out == "The answer is 42 [1]."
    assert len(llm.calls) == 1
    msgs = llm.calls[0]
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == SYSTEM_PROMPT
    assert isinstance(msgs[1], HumanMessage)
    assert "doc.pdf" in msgs[1].content and "page 5" in msgs[1].content
    assert "What is the answer?" in msgs[1].content


def test_connection_error_wrapped_in_llm_init_error():
    class DownLLM:
        def invoke(self, _):
            raise ConnectionError("connection refused")

    with pytest.raises(LLMInitError, match="Could not reach"):
        generate_answer(DownLLM(), "q", [chunk()])


# --- get_llm / check_ollama --------------------------------------------------------

def test_get_llm_ollama():
    from langchain_ollama import ChatOllama

    llm = get_llm(get_settings(llm_provider="ollama", llm_model="llama3.2:3b"))
    assert isinstance(llm, ChatOllama)
    assert llm.model == "llama3.2:3b"


def test_get_llm_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMInitError, match="OPENAI_API_KEY"):
        get_llm(get_settings(llm_provider="openai", llm_model="gpt-4o-mini"))


def test_get_llm_openai_with_key(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    llm = get_llm(get_settings(llm_provider="openai", llm_model="gpt-4o-mini"))
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.openai_api_base == "http://localhost:9999/v1"


def test_check_ollama_unreachable():
    ok, msg = check_ollama(get_settings(ollama_base_url="http://127.0.0.1:1"), timeout=0.5)
    assert ok is False
    assert "ollama serve" in msg


class _FakeResp:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_check_ollama_model_missing_and_present(monkeypatch):
    body = b'{"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b"}]}'
    monkeypatch.setattr(generation.urllib.request, "urlopen", lambda *a, **k: _FakeResp(body))
    ok, msg = check_ollama(get_settings(llm_model="llama3.2:3b"))
    assert ok and "available" in msg
    ok, msg = check_ollama(get_settings(llm_model="mistral"))
    assert not ok and "ollama pull mistral" in msg


def test_system_prompt_has_number_and_table_rule():
    # Regression for the BLEU 38.1-vs-28.4 error: flattened tables must not be read blindly.
    assert "Copy numbers exactly" in SYSTEM_PROMPT
    assert "table" in SYSTEM_PROMPT.lower()


def test_ollama_response_error_wrapped_with_pull_hint():
    class ResponseError(Exception):  # same name as ollama._types.ResponseError
        pass

    class BrokenLLM:
        def invoke(self, messages):
            raise ResponseError("model 'llama9' not found")

    chunks = [RetrievedChunk(text="t", source="a.pdf", page=1, score=0.9, chunk_id="c")]
    with pytest.raises(LLMInitError, match="ollama pull"):
        generate_answer(BrokenLLM(), "q?", chunks)


def test_system_prompt_has_false_premise_rule():
    # Regression for A2 ("Why does SBERT use [CLS] as default?"): the model must not justify a premise
    # that no passage states.
    assert "do not explain or justify it" in SYSTEM_PROMPT


def test_get_llm_passes_sampling_and_context_settings(monkeypatch):
    from langchain_ollama import ChatOllama

    llm = get_llm(get_settings(llm_provider="ollama", llm_temperature=0.2, llm_top_p=0.8, llm_top_k=20, llm_num_ctx=8192))
    assert isinstance(llm, ChatOllama)
    assert (llm.temperature, llm.top_p, llm.top_k, llm.num_ctx) == (0.2, 0.8, 20, 8192)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    api = get_llm(get_settings(llm_provider="openai", llm_model="gpt-4o-mini", llm_top_p=0.7))
    assert api.top_p == 0.7
