"""元ROOTの再束縛を、実起動path認証の迂回路にしない。子の再走は不要。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pytest
import parent_client_v2 as V


def test_old_root_does_not_redirect(monkeypatch: Any, tmp_path: Path) -> None:
    expected = V.worker_path()
    monkeypatch.setattr(V.T, 'ROOT', tmp_path)
    assert V.worker_path() == expected


def test_unpinned_worker_directory_rejected(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(V, 'PARENT', tmp_path)
    with pytest.raises(ValueError, match='runtime_worker_path_pin'):
        V.worker_path()
