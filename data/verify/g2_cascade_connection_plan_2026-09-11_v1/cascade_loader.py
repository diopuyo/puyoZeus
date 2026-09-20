"""原factoryで生成された同一belief/Registryに既存cascade v2を接続する。"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent.parent / 'g2_basis_cascade_candidate_2026-09-11_v1'


def wrap(original: Any, load: Any) -> Any:
    def modules() -> Any:
        parts = original()
        cascade = load('_g2_real_basis_cascade_math', ROOT / 'cascade.py', {'belief': parts.belief})
        first = load('_g2_real_basis_cascade_mode', ROOT / 'mode.py',
                     {'physical_tracking': parts.mode, 'cascade': cascade})
        second = load('_g2_real_basis_cascade_mode_v2', ROOT / 'mode_v2.py', {'mode': first})
        return N(**(vars(parts) | dict(mode=second)))
    return modules
