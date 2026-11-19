# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.vdb.postgres import (
    _DEFAULT_VECTOR_SIZE,
    PostgresVDB,
    _default_vector_size,
)

_URL = "postgresql://user:pw@localhost:5432/db"


def test_default_vector_size_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_VECTOR_SIZE", raising=False)
    assert _default_vector_size() == _DEFAULT_VECTOR_SIZE


def test_default_vector_size_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", " 3072 ")
    assert _default_vector_size() == 3072


def test_default_vector_size_blank_env_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", "")
    assert _default_vector_size() == _DEFAULT_VECTOR_SIZE


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "1.5"])
def test_default_vector_size_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", raw)
    with pytest.raises(ValueError, match="EMBED_VECTOR_SIZE"):
        _default_vector_size()


def test_constructor_picks_up_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", "3072")
    assert PostgresVDB(connection_string=_URL).vector_size == 3072


def test_constructor_explicit_vector_size_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", "3072")
    assert PostgresVDB(connection_string=_URL, vector_size=1024).vector_size == 1024


def _vdb_with_stored_width(stored: int) -> tuple[PostgresVDB, MagicMock]:
    vdb = PostgresVDB(connection_string=_URL)
    connect = MagicMock()
    cur = connect.return_value.__enter__.return_value.cursor.return_value.__enter__
    cur.return_value.fetchone.return_value = (stored,)
    return vdb, connect


def test_stored_width_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", "3072")
    vdb, connect = _vdb_with_stored_width(2048)
    with patch("auto_ontology.vdb.postgres.psycopg.connect", connect):
        with pytest.raises(ValueError, match="EMBED_VECTOR_SIZE") as exc:
            vdb._check_stored_width()
    assert "3072" in str(exc.value) and "2048" in str(exc.value)
    assert "public.data_objects_layer" in str(exc.value)


def test_stored_width_match_passes_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", "3072")
    vdb, connect = _vdb_with_stored_width(3072)
    with patch("auto_ontology.vdb.postgres.psycopg.connect", connect):
        vdb._check_stored_width()
        vdb._check_stored_width()
    assert connect.call_count == 1


def test_untyped_column_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_VECTOR_SIZE", "3072")
    vdb, connect = _vdb_with_stored_width(-1)
    with patch("auto_ontology.vdb.postgres.psycopg.connect", connect):
        vdb._check_stored_width()
