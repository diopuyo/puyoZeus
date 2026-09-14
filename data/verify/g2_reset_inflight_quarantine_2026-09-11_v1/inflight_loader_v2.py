"""旧ローダー・成果物を保持したまま修正版だけを専用aliasへ接続する。"""
from __future__ import annotations
from typing import Any
import inflight_loader as OLD

ROOT, load = OLD.ROOT, OLD.load
GUARD = load('_g2_inflight_guard_v2', 'quarantine_v2.py', {'quarantine': OLD.GUARD})
CONNECTION = load('_g2_inflight_connection_v2', 'connection.py', {'_g2_inflight_guard': GUARD})


def basis_connection() -> Any:
    root = ROOT.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1'
    previous = OLD.basis_connection()
    gate = load('_g2_settled_basis_v3', root / 'gate_v3.py', {'gate_v2': previous.G})
    return load('_g2_settled_basis_actual_v2', root / 'actual_connection_v2.py',
                {'actual_connection': previous, 'gate_v3': gate})
