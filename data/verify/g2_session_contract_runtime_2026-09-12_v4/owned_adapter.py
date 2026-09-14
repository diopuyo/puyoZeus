"""旧whole実入口の二つの依存選択だけ替える。旧コード/旧成果物は保持する。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
from types import FunctionType
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT.parent
PRIOR = VERIFY / 'g2_live_probability_context_2026-09-12_v1'
START = VERIFY / 'g2_start_client_session_contract_2026-09-12_v1'
SIDE = VERIFY / 'g2_historical_chain_next_reset_2026-09-12_v1'
sys.path.insert(0, str(PRIOR))
import whole_live_adapter as A

protect = A.protect

SIDE_ALIAS = '_g2_historical_chain_next_actual_connection'
OWNED_ALIAS = '_g2_live_probability_owned_loader'
DEPENDENCY_ROOTS = (
    'g2_reset_inflight_quarantine_2026-09-11_v1',
    'g2_hidden_basis_initialization_2026-09-11_v1',
    'g2_cascade_connection_plan_2026-09-11_v1',
    'g2_side_occurrence_retirement_2026-09-11_v1',
    'g2_tracker_snapshot_repair_2026-09-11_v1',
    'g2_archive_close_boundary_2026-09-11_v1',
    'g2_transition_endpoint_votes_2026-09-11_v1',
    'g2_basis_cascade_candidate_2026-09-11_v1',
    'g2_reset_settled_basis_gate_2026-09-11_v1',
    'g2_probabilistic_scope_candidate_2026-09-11_v1',
    'g2_belief_hidden_landing_2026-09-11_v1',
)


def require_owned(owner: Any) -> None:
    if sys.modules.get(OWNED_ALIAS) is not owner or Path(owner.__file__).resolve() != PRIOR / 'loader.py':
        raise RuntimeError('owned_loader_identity')


def replace_owned(stack: Any, module: Any, name: str, value: Any) -> None:
    previous = getattr(module, name)
    setattr(module, name, value)
    def restore(kind: Any, body: Any, trace: Any) -> bool:
        if getattr(module, name) is value:
            setattr(module, name, previous)
        elif body is None:
            raise RuntimeError('repair_dependency_foreign:' + name)
        return False
    stack.push(restore)


def sources() -> tuple[Path, ...]:
    start = tuple(START / name for name in ('start_trace.py', 'start_qualification.py', 'start_capture.py',
                                          'start_join.py', 'start_worker.py', 'start_client.py'))
    side = tuple(SIDE / name for name in ('side_reset_op.py', 'selected_side_lease.py', 'side_actual_connection.py'))
    dependencies = tuple(p for name in DEPENDENCY_ROOTS for p in (VERIFY / name).glob('*.py'))
    return tuple(sorted(set(start + side + dependencies + (ROOT / 'owned_adapter.py',))
                        | set(START.glob('*.py'))))


def install_side(stack: Any, owner: Any) -> None:
    original = owner.dependencies
    owned: dict[str, Any] = {}
    if SIDE_ALIAS in sys.modules:
        raise RuntimeError('repair_side_foreign_alias')
    def dependencies() -> Any:
        require_owned(owner)
        deps = original()
        loader = owner.bootstrap()
        existing = sys.modules.get(SIDE_ALIAS)
        if SIDE_ALIAS in owned and existing is not owned[SIDE_ALIAS]:
            raise RuntimeError('repair_side_replaced')
        if existing is not None and owned.get(SIDE_ALIAS) is not existing:
            raise RuntimeError('repair_side_replaced')
        try:
            side = loader.load(SIDE_ALIAS, SIDE / 'side_actual_connection.py', {'inflight_loader': loader})
        finally:
            partial = sys.modules.get(SIDE_ALIAS)
            if existing is None and partial is not None and getattr(partial, '__file__', None) == str(SIDE / 'side_actual_connection.py'):
                owned[SIDE_ALIAS] = partial
        return replace(deps, side=side)
    def release(kind: Any, body: Any, trace: Any) -> bool:
        for name, module in owned.items():
            if sys.modules.get(name) is module:
                sys.modules.pop(name)
            elif name in sys.modules and body is None:
                raise RuntimeError('repair_side_alias_restore')
        return False
    stack.push(release)
    replace_owned(stack, owner, 'dependencies', dependencies)


def configured(stack: Any) -> Any:
    namespace = dict(vars(A.D), ROOT=START)
    for name in ('observer_and_anchor', 'session_creator'):
        original = getattr(A.D, name)
        selected = FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__, original.__closure__)
        selected.__kwdefaults__ = original.__kwdefaults__
        replace_owned(stack, A.D, name, selected)
    original_sources = A.D.source_paths
    replace_owned(stack, A.D, 'source_paths', lambda: tuple(sorted(set(original_sources()) | set(sources()))))
    selected = A.configured(stack)
    owner = sys.modules[OWNED_ALIAS]
    require_owned(owner)
    original_build = owner.build
    if original_build.__globals__ is not vars(owner):
        raise RuntimeError('owned_build_globals')
    install_side(stack, owner)
    selected_dependencies = owner.dependencies
    def build(parent: type, *, end_frame: int) -> type:
        require_owned(owner)
        if owner.dependencies is not selected_dependencies:
            raise RuntimeError('owned_dependencies_changed')
        return original_build(parent, end_frame=end_frame)
    replace_owned(stack, owner, 'build', build)
    return selected
