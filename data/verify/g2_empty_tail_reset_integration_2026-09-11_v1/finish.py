"""資格追加が原76の最終検収を壊していないことを原finishで検査する。"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Any


def scoped(kept: Any, Q: Any) -> None:
    import empty_reset as E
    import runtime_stage as S
    retired,world = Q.KEPT['retired_samecall'],Q.KEPT['reset_world']
    assert E.digest(retired)==Q.KEPT['retired_samecall_sha'] and E.digest(world)==Q.KEPT['reset_world_sha']
    assert retired['retired_owner_verified'] and not retired['current_history_replaced']
    assert world['world']['world_PB_verified'] and world['actual_factory']
    stage = S.verify(kept)
    kept['runtime'].write(kept['state']['output']/'EMPTY_RESET_RUNTIME.json',
        dict(continuation=Q.KEPT['reset_continuation'],retired_samecall=retired,world=world,stage=stage,
            multi_scope_finalizer_verified=True,independent_review_pending=True,
            runtime_finalization_allowed=False,GPU_GO=False,quality_gate_clear=False,
            actual_factory=True,artificial_inputs=True,physical_verified=False))


def finish(kept: Any) -> None:
    import run_qualification as Q
    if 'reset_continuation' in Q.KEPT:
        # 実reset継続を検査しても、旧単scopeの世界最終検査へ偽の現役bindingを渡さない。
        assert Q.KEPT['reset_continuation']['reset_executed']
        assert not Q.KEPT['reset_continuation']['multi_scope_finalizer_verified']
        return scoped(kept,Q)
    path = Path(__file__).resolve().parent.parent/'g2_empty_tail_finalizer_2026-09-10_v1/finish.py'
    spec = importlib.util.spec_from_file_location('_empty_reset_original_finish', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.finish(kept)
