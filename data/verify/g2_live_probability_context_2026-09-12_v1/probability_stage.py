"""確率移行前の実decision範囲で、元goals/全J/firingの構造検査を維持する。"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import sys
from typing import Any
import stage_retired_identity as IDENTITY


def derive(original: Any, S: Any, stage: Any, lib: Any, evidence: Any,
           rows: Any, lease: Any, runtime: Any = None) -> Any:
    """整数の新baselineを作らず、認証済み旧scopeにだけAuditを束縛する。"""
    path = original.STAGE.ADAPTER
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original.STAGE.ADAPTER_SHA
    adapter = original.Q.module('_probability_original_stage_adapter', path)
    old = lease.archive.binding.owner.state
    frames = [row['scope']['frame_idx'] for row in rows]
    assert frames == list(range(old.baseline_through.frame, lease.empty_evidence.frame + original.Q.E.STRIDE,
                                original.Q.E.STRIDE)), 'probability_stage_old_coverage'
    scope = asdict(old.scope)
    assert all(row['decision']['history_state']['scope'] == scope for row in rows)
    value = S.derive(stage, lib, evidence)
    assert runtime is not None, 'stage_actual_runtime_required'
    return adapter.clone(value,
        live_identity=IDENTITY.adapted(value.__globals__['live_identity'], runtime, lease, original.Q),
        initializer=adapter.initializer(value.__globals__['initializer'], dict.fromkeys(frames, scope)),
        transformed=adapter.transformed(value.__globals__['transformed'], frames))


def actual_factory(state: Any, supplied: Any = None) -> Any:
    """CPUは実参照を明示輸送する。live constructor資格をstateへ追加しない。"""
    if supplied is None:
        return state['private_suffix_factory']
    connection = state['probabilistic_basis_connection']
    assert supplied is connection.recovery.factory, 'stage_actual_factory'
    assert state['probabilistic_tracking_mode'].connection is connection, 'stage_actual_connection'
    if 'private_suffix_factory' in state:
        assert state['private_suffix_factory'] is supplied, 'stage_live_factory_conflict'
    return supplied


def verify(original: Any, goals: Any, supplied: Any, legal: Any, output: Any,
           state: Any, boundary: Any, *, firing_rows: Any, factory: Any = None) -> dict[str, Any]:
    factory = actual_factory(state, factory)
    assert factory.controller.legal is legal and output == state['output']
    lease = state['repeat_scope_guard'].reset_lease
    full = original.read(output, 'directional_history.jsonl')
    journal = original.read(output, 'atomic_journal.jsonl')
    rows = [row for row in full if 'decision' in row]
    assert supplied == full or supplied == rows, 'actual_supplied_history'
    evidence = original.read(output, 'EMPTY_TAIL_LIVE_EVIDENCE.json')
    with ExitStack() as stack:
        previous = list(sys.path)
        stack.callback(sys.path.__setitem__, slice(None), previous)
        sys.path[:0] = [str(original.Q.FINAL), str(original.STAGE.PRIVATE), str(original.STAGE.LINK)]
        S = original.Q.module('_probability_finish_stage1', original.Q.FINAL / 'empty_stage1.py')
        join = original.Q.module('_probability_finish_join', original.STAGE.LIVE / 'prepop_join.py')
        with S.libraries() as lib, lib.F.session() as stage:
            function = derive(original, S, stage, lib, evidence, rows, lease, runtime=state)
            result = function(goals, rows, legal, output, firing_rows=firing_rows, journal_rows=journal,
                conditional_rows=state['conditional_full_rows'], controller=factory.controller, factory=factory)
            saved = lib.L.check(rows, journal)
        joined = join.check(factory.controller.hidden_rolling_prepop_checks, saved, rows)
    assert result['same_live_controller_verified'] and joined['rolling_exercised']
    assert result['issued'] == len(boundary['issued_frames']) and result['released'] == 0
    return dict(stage1=result, rolling_prepop_join=joined, actual_goals_passthrough=True,
                actual_firing_rows_passthrough=True, quality_gate_clear=False)
