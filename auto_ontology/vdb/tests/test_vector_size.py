# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_ontology.vdb.postgres import _DEFAULT_VECTOR_SIZE, _default_vector_size


def test_default_vector_size_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_DIMENSIONS", raising=False)
    assert _default_vector_size() == _DEFAULT_VECTOR_SIZE


def test_default_vector_size_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_DIMENSIONS", " 3072 ")
    assert _default_vector_size() == 3072


def test_default_vector_size_blank_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_DIMENSIONS", "")
    assert _default_vector_size() == _DEFAULT_VECTOR_SIZE


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "1.5"])
def test_default_vector_size_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("EMBED_DIMENSIONS", raw)
    with pytest.raises(ValueError, match="EMBED_DIMENSIONS"):
        _default_vector_size()
