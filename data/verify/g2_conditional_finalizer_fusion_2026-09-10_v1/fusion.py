"""独立二段検査を結合し、元整数集計を維持する。"""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent / 'g2_conditional_finalizer_compatibility_2026-09-10_v1'
WORLD = ROOT.parent / 'g2_conditional_finalizer_world_2026-09-10_v1'


@contextmanager
def modules() -> Iterator[Any]:
    previous, alias = list(sys.path), '_fused_conditional_stage1'
    old = sys.modules.get(alias)
    try:
        sys.path[:0] = [str(STAGE1), str(WORLD)]
        spec = importlib.util.spec_from_file_location(alias, STAGE1 / 'compat.py')
        stage1 = importlib.util.module_from_spec(spec)
        sys.modules[alias] = stage1
        spec.loader.exec_module(stage1)
        import world_verify as world
        assert Path(world.__file__).resolve() == WORLD / 'world_verify.py'
        yield stage1, world
    finally:
        sys.path[:] = previous
        if old is None:
            sys.modules.pop(alias, None)
        else:
            sys.modules[alias] = old


def audit(goals: Any, rows: Any, legal: Any, output: Any, *, firing_rows: Any,
          journal_rows: Any, conditional_rows: Any, hidden_events: Any,
          hidden_history: Any, hidden_lifetime: Any, outer_rows: Any,
          controller: Any = None, factory: Any = None) -> dict[str, Any]:
    with modules() as (stage1, world):
        if stage1.conditional(rows):
            # 原legalの遅延importより先に、元worldの固定backendを解決する。
            with world.G.libraries():
                pass
        result = stage1.audit_stage1(goals, rows, legal, output, firing_rows=firing_rows,
            journal_rows=journal_rows, conditional_rows=conditional_rows, controller=controller, factory=factory)
        if not stage1.conditional(rows):
            stage1.F.require(not hidden_events and not hidden_history and not hidden_lifetime,
                'unexpected_hidden_normal')
            return result
        checked = world.verify_world(history_rows=rows, journal_rows=journal_rows,
            conditional_rows=conditional_rows, hidden_events=hidden_events, hidden_history=hidden_history,
            hidden_lifetime=hidden_lifetime, outer_rows=outer_rows() if callable(outer_rows) else outer_rows,
            controller=controller, factory=factory)
        live = result['same_live_controller_verified'] and checked['actual_live_scope_verified']
        return result | dict(conditional_world=checked, world_PB_verified=checked['world_PB_verified'],
            runtime_finalization_allowed=live, conditional_current_counted_as_integer=False,
            quality_gate_clear=False, production_permission=False, physical_certified=False)


def evaluate(goals: Any, rows: Any, legal: Any, output: Any, *, firing_rows: Any,
             journal_rows: Any, conditional_rows: Any, hidden_events: Any,
             hidden_history: Any, hidden_lifetime: Any, outer_rows: Any,
             controller: Any, factory: Any) -> dict[str, Any]:
    result = audit(goals, rows, legal, output, firing_rows=firing_rows, journal_rows=journal_rows,
        conditional_rows=conditional_rows, hidden_events=hidden_events, hidden_history=hidden_history,
        hidden_lifetime=hidden_lifetime, outer_rows=outer_rows, controller=controller, factory=factory)
    with modules() as (stage1, world):
        stage1.F.require(not stage1.conditional(rows) or result['runtime_finalization_allowed'],
            'actual_conditional_runtime_required')
    return result
