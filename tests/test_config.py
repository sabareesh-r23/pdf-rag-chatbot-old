"""Validation of the retrieval and LLM parameter settings."""
from __future__ import annotations

import pytest

from rag.config import get_settings


def test_defaults():
    s = get_settings()
    assert s.retrieval_mode in {"similarity", "mmr", "hybrid"}
    assert 0 <= s.mmr_lambda <= 1 and s.mmr_fetch_k > 0
    assert 0 < s.llm_top_p <= 1 and s.llm_num_ctx >= 512


@pytest.mark.parametrize(
    "overrides",
    [
        {"retrieval_mode": "keyword-only"},
        {"mmr_fetch_k": 0},
        {"mmr_lambda": 1.5},
        {"llm_temperature": -0.1},
        {"llm_top_p": 0.0},
        {"llm_num_ctx": 100},
    ],
)
def test_invalid_values_rejected(overrides):
    with pytest.raises(ValueError):
        get_settings(**overrides)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "MMR")
    monkeypatch.setenv("MMR_LAMBDA", "0.7")
    monkeypatch.setenv("LLM_NUM_CTX", "8192")
    s = get_settings()
    assert (s.retrieval_mode, s.mmr_lambda, s.llm_num_ctx) == ("mmr", 0.7, 8192)
