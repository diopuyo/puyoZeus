"""確率追跡候補を原factory生成後だけ、衝突しない専用名で読み込む。"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import inflight_loader_v2 as L
import basis_binding_loader as PREVIOUS

ROOT = L.ROOT
GUARD = L.load('_g2_inflight_guard_v3', 'quarantine_v3.py', {'quarantine_v2': L.GUARD})
QUALIFIED = L.load('_g2_inflight_guard_v4', 'quarantine_v4.py', {'quarantine_v3': GUARD})
CONNECTION = L.load('_g2_inflight_connection_v4', 'connection.py', {'_g2_inflight_guard': QUALIFIED})


def modules() -> Any:
    root = ROOT.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1'
    math_root = ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'
    old = PREVIOUS.connection()
    v4 = L.load('_g2_settled_basis_v4', root / 'gate_v4.py', {'gate_v3': L.basis_connection().G})
    actual = L.load('_g2_tracking_basis_observer', root / 'actual_connection_v2.py',
                    {'actual_connection': L.OLD.basis_connection(), 'gate_v3': v4})
    binding = L.load('_g2_tracking_basis_binding', root / 'basis_registry_connection.py',
        dict(belief=old.B, conditioning=old.C, registry=old.R, serialization=old.S, actual_connection_v2=actual))
    native = L.load('_g2_tracking_native', math_root / 'native_consumption.py', {'belief': old.B})
    base_mode = L.load('_g2_tracking_mode', root / 'tracking_mode.py', {'native_consumption': native})
    connected_mode = L.load('_g2_tracking_mode_v2', root / 'tracking_mode_v2.py', {'tracking_mode': base_mode})
    placement_root = ROOT.parent / 'g2_belief_hidden_landing_2026-09-11_v1'
    placement = L.load('_g2_tracking_placement', placement_root / 'frozen_placement.py')
    hidden = L.load('_g2_tracking_hidden_landing', placement_root / 'hidden_landing.py',
                    {'belief': old.B, 'frozen_placement': placement})
    transition = L.load('_g2_tracking_transition', math_root / 'transition_candidates.py',
                        {'belief': old.B, 'hidden_landing': hidden})
    physical = L.load('_g2_tracking_physical', root / 'physical_tracking.py',
                        {'tracking_mode': connected_mode, 'transition_candidates': transition, 'serialization': old.S})
    lease = L.load('_g2_tracking_lease', root / 'tracking_lease.py', {'tracking_mode': base_mode})
    return SimpleNamespace(actual=actual, binding=binding, mode=physical, lease=lease, belief=old.B)
