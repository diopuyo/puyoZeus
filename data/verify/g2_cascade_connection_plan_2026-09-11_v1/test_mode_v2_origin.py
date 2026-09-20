"""v11の二重拒否仮説を既存mode_v2と既存人工Jで反証。実SM接続ではない。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent / 'g2_basis_cascade_candidate_2026-09-11_v1'
sys.path.insert(0, str(ROOT))
import test_cascade as F
spec = importlib.util.spec_from_file_location('_g2_mode_v2_origin_recheck', ROOT / 'mode_v2.py')
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def test_basis_origin_without_native_passes_existing_mode_v2(monkeypatch: Any) -> None:
    monkeypatch.setattr(F.C, 'Mode', M.Mode)
    F.test_basis_chain_then_next_hand_keeps_distribution()
