"""CPU実kept参照から原goals/全J/公開境界を構造検査へ渡す。live資格は作らない。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import probability_boundary as B
import probability_saved as SAVED
import probability_stage as S

BASE = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1'


def boundary(state: Any, lease: Any) -> dict[str, Any]:
    output = state['output']
    return B.check(state['postcommit_publication_consumer'].rows,
        SAVED.read(output, 'directional_history.jsonl'), SAVED.read(output, 'atomic_journal.jsonl'),
        lease.recovery.rows, SAVED.read(output, 'PROBABILISTIC_TRACKING.jsonl'),
        history_first=lease.archive.binding.owner.state.baseline_through.frame,
        end_frame=state['probabilistic_tracking_mode'].native.last_frame,
        proof_frame=lease.empty_evidence.frame,
        basis_frame=state['probabilistic_tracking_mode'].activation['frame'],
        scope=state['probabilistic_basis_connection'].binding.scope)


def verify(kept: dict[str, Any]) -> dict[str, Any]:
    state, factory = kept['state'], kept['factory']
    before = dict(state)
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), previous)
        sys.path.insert(0, str(BASE))
        original = importlib.import_module('runtime_stage')
        assert Path(original.__file__).resolve() == BASE / 'runtime_stage.py'
        captured = original.Q.KEPT['runtime']
        assert captured['factory'] is factory and captured['state'] is state, 'stage_original_kept'
        runtime = captured['runtime']
        sys.path.insert(0, str(original.PRIVATE))
        helper = runtime.load('_probability_stage_probe_goals_helper', original.PRIVATE / 'fusion.py', stack)
        full = SAVED.read(state['output'], 'directional_history.jsonl')
        rows = [row for row in full if 'decision' in row]
        outer = state['postcommit_publication_consumer'].rows
        goals = helper.goals(runtime, stack, rows, outer)
        joined = boundary(state, state['repeat_scope_guard'].reset_lease)
        # 元CPU finalizer同様に発火列は空。実firing constructorの検証とは区別する。
        value = S.verify(N(Q=original.Q, STAGE=original, read=SAVED.read), goals, full,
            factory.controller.legal, state['output'], state, joined, firing_rows=[], factory=factory)
    assert set(state) == set(before) and all(state[k] is v for k, v in before.items())
    return dict(structure=value, boundary=joined, actual_factory=True,
                actual_firing_constructor=False, stage1_verified=True,
                full_probability_finalizer_verified=False, actual_video=False, quality_gate_clear=False)
