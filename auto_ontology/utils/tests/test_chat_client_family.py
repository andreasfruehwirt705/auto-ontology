# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.utils import llm_invoke


def test_family_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAT_CLIENT", raising=False)
    assert llm_invoke._chat_client_family() == "auto"


@pytest.mark.parametrize(
    "raw, expected", [(" OpenAI ", "openai"), ("nvidia", "nvidia"), ("", "auto")]
)
def test_family_is_normalised(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv("CHAT_CLIENT", raw)
    assert llm_invoke._chat_client_family() == expected


def test_family_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_CLIENT", "anthropic")
    with pytest.raises(ValueError, match="CHAT_CLIENT"):
        llm_invoke._chat_client_family()


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
