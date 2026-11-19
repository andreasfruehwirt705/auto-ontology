# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from auto_ontology.utils import llm_invoke


_VARS = ("DEFAULT_MODELS_CLIENT", "REASONING_CLIENT", "NON_REASONING_CLIENT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)


def test_family_defaults_to_auto() -> None:
    assert llm_invoke._chat_client_family("REASONING") == "auto"


@pytest.mark.parametrize(
    "raw, expected", [(" OpenAI ", "openai"), ("nvidia", "nvidia"), ("", "auto")]
)
def test_family_is_normalised(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv("REASONING_CLIENT", raw)
    assert llm_invoke._chat_client_family("REASONING") == expected


def test_family_falls_back_to_default_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_MODELS_CLIENT", "openai")
    assert llm_invoke._chat_client_family("REASONING") == "openai"
    assert llm_invoke._chat_client_family("NON_REASONING") == "openai"
    monkeypatch.setenv("NON_REASONING_CLIENT", "nvidia")
    assert llm_invoke._chat_client_family("REASONING") == "openai"
    assert llm_invoke._chat_client_family("NON_REASONING") == "nvidia"


def test_family_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NON_REASONING_CLIENT", "anthropic")
    with pytest.raises(ValueError, match="NON_REASONING_CLIENT"):
        llm_invoke._chat_client_family("NON_REASONING")


@pytest.mark.parametrize(
    "model, family, expected",
    [
        ("azure/anthropic/claude-opus-5-5", "auto", True),
        ("openai/gpt-4.1", "auto", True),
        ("aws/some-model", "auto", True),
        ("nvidia/nemotron-3.5-lightning-30b-a3b", "auto", False),
        ("gpt-6.2-sol", "auto", False),
        ("gpt-6.2-sol", "openai", True),
        ("nvidia/nemotron-3.5-lightning-30b-a3b", "openai", True),
        ("azure/anthropic/claude-opus-5-5", "nvidia", False),
    ],
)
def test_use_openai_client(model: str, family: str, expected: bool) -> None:
    assert llm_invoke._use_openai_client(model, family) is expected


def _build(family: str) -> object:
    return llm_invoke._build_client(
        model="gpt-x",
        api_key="k",
        base_url="http://x",
        temperature=0,
        max_tokens=8,
        family=family,
    )


def test_build_client_openai_family() -> None:
    assert isinstance(_build("openai"), ChatOpenAI)


def test_build_client_nvidia_family() -> None:
    assert isinstance(_build("nvidia"), ChatNVIDIA)


def test_get_llm_client_honours_reasoning_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_invoke, "_API_KEY", "k")
    monkeypatch.setattr(llm_invoke, "_MODEL_NAME", "gpt-x")
    monkeypatch.setattr(llm_invoke, "_BASE_URL", "http://x")
    assert isinstance(llm_invoke.get_llm_client(), ChatNVIDIA)
    monkeypatch.setenv("REASONING_CLIENT", "openai")
    assert isinstance(llm_invoke.get_llm_client(), ChatOpenAI)
