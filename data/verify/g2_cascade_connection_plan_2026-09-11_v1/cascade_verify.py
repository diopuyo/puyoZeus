"""原基準取得と実Jの基準連鎖一回だけを検収する。新手/公開の合格とは分ける。"""
from __future__ import annotations
from dataclasses import asdict
import json
from typing import Any
import cascade_inputs as I

PREFIX_COUNT = 80
UPDATES = PREFIX_COUNT + I.POST_COUNT
J_STEPS = UPDATES * 2


def verify(recovery: Any, lease: Any, factory: Any, state: Any, trace: Any, prior: Any) -> Any:
    mode, connection = state['probabilistic_tracking_mode'], state['probabilistic_basis_connection']
    assert mode.activation is not None and mode.error is None and mode.native is not None
    assert mode.basis_cascade_closed and mode.basis_origin is not None
    assert len(mode.applied) == 1 and not mode.native.pending and not mode.native.seen_occurrences
    receipt = mode.applied[0]
    assert receipt['kind'] == 'basis_cascade' and receipt['next_consumed'] is False
    assert 0 < receipt['distribution_report']['retained_mass'] <= 1
    current = connection.registry.current(connection.binding)
    assert current.frame == receipt['applied_frame'] > mode.activation['frame']
    lease.archive.verify()
    old, saved, record = recovery.archive[0]
    assert old.owner.state is saved and asdict(saved) == record['owner']
    assert recovery.reset_count == 1 and recovery.baseline_count == 0 and recovery.pending is None
    assert factory.controller.history.get('1P') is None
    assert factory.provider.journal.steps == J_STEPS and len(trace) == I.POST_COUNT
    assert recovery.pipe._active_chain_1p is None and recovery.pipe._sm_1p.context.state.value == 'stable'
    retired = state['basis_baseline_retirement']
    assert retired.used and len([row for row in retired.rows if row['suppressed']]) == 1
    receiver = state['postcommit_current_receiver']
    assert not receiver.errors and receiver.released == prior['released']
    votes = state['qualified_prior_votes']
    assert votes.failure is None and votes.rows and votes.allowed.error is None
    result = dict(basis_only=False, basis_cascade_only=True, actual_J_steps=J_STEPS,
        probabilistic_basis_frame=mode.activation['frame'], physical_transitions=1,
        native_consumptions=0, old_archive_unchanged=True, qualified_votes=len(votes.rows),
        hidden_basis_connected=True, artificial_inputs=True, actual_video_run=False,
        integer_continuation_verified=False, full_finalizer_verified=False, quality_gate_clear=False)
    with (state['output'] / 'PROBABILISTIC_TARGET_RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    with (state['output'] / 'BASIS_CASCADE_FINAL.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(transition=receipt, active_origin=False,
            final_time=trace[-1]['frame']/60, chain_until=recovery.pipe._chain_until_1p,
            final_state='stable', quality_gate_clear=False), stream, indent=2)
    return result
