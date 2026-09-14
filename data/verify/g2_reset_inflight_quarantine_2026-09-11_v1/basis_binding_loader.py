"""原factory生成後に確率所有権の依存だけを専用名で読み込む。"""
from __future__ import annotations
from typing import Any
import inflight_loader_v2 as L


def connection() -> Any:
    root = L.ROOT.parent / 'g2_probabilistic_scope_candidate_2026-09-11_v1'
    physics = L.load('_g2_basis_binding_physics', root / 'frozen_physics.py')
    belief = L.load('_g2_basis_binding_belief', root / 'belief.py', {'frozen_physics': physics})
    modules = {name: L.load('_g2_basis_binding_' + name, root / (name + '.py'), {'belief': belief})
               for name in ('conditioning', 'registry', 'serialization')}
    modules.update(belief=belief, actual_connection_v2=L.basis_connection())
    path = L.ROOT.parent / 'g2_reset_settled_basis_gate_2026-09-11_v1/basis_registry_connection.py'
    return L.load('_g2_basis_registry_actual', path, modules)
