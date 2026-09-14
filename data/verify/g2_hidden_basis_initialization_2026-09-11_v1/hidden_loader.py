"""原factory生成後のローダーで、隠し基準gate/保存票だけを専用aliasへ差替える。"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import tracking_loader as OLD

ROOT, CONNECTION = OLD.ROOT, OLD.CONNECTION
PRIVATE = Path(__file__).resolve().parent


def modules() -> Any:
    previous = OLD.modules()
    loader = OLD.L.load
    base = ROOT.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1'
    gate = loader('_g2_hidden_initialization_gate', PRIVATE / 'hidden_basis_gate.py',
                  {'gate_v4': previous.actual.G})
    actual = loader('_g2_hidden_initialization_observer', base / 'actual_connection_v2.py',
                    {'actual_connection': previous.actual.OLD, 'gate_v3': gate})
    binding = loader('_g2_hidden_initialization_binding_base', base / 'basis_registry_connection.py',
        dict(belief=previous.binding.B, conditioning=previous.binding.C,
             registry=previous.binding.R, serialization=previous.binding.S, actual_connection_v2=actual))
    updated = loader('_g2_hidden_initialization_binding', PRIVATE / 'basis_registry_v2.py',
                     {'basis_registry_connection': binding})
    return SimpleNamespace(**(vars(previous) | dict(actual=actual, binding=updated)))
