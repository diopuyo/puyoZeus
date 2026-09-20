"""v35の型定義解放後の失敗を元Archive/Basisで再現する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'g2_empty_tail_archive_candidate_2026-09-10_v1'))
spec = importlib.util.spec_from_file_location('_g2_close_original_fixture',
    ROOT / 'g2_empty_tail_archive_candidate_2026-09-10_v1/test_empty_archive.py')
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)
modules, fixture, actual = F.modules, F.fixture, F.actual


def test_archive_passes_before_module_release_and_fails_after(actual: Any) -> None:
    archive = F.X.Archive(actual.local.controller, actual.binding)
    archive.verify()
    name = type(actual.basis).__module__
    original = sys.modules.pop(name)
    try:
        with pytest.raises(AssertionError, match='empty_archive_basis_type'):
            archive.verify()
    finally:
        sys.modules[name] = original
    archive.verify()
