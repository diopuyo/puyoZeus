"""連鎖一回の保存→実Registry→元archive寿命検査。元基準onlyの成功には偽装しない。"""
from __future__ import annotations
from dataclasses import asdict
from types import FunctionType
from typing import Any
import probabilistic_finish as P
import finish_v2 as A
import cascade_verify as V

EVIDENCE: dict[str, Any] = {}


def saved(output: Any, serializer: Any) -> tuple[Any, Any]:
    result, target = P.read(output, 'RESULT.json'), P.read(output, 'PROBABILISTIC_TARGET_RESULT.json')
    P.require(result['exit_code'] == 0 and result['updates'] == V.UPDATES and result['J_steps'] == V.J_STEPS
              and result['guards_unchanged'] is True and result['gpu'] is False, 'cascade_body')
    canonical = result['reset_continuation']
    P.require(P.TARGET_KEYS <= target.keys() and all(canonical[k] == v for k, v in target.items()), 'cascade_target')
    P.require(target['basis_only'] is False and target['basis_cascade_only'] is True
              and target['actual_J_steps'] == V.J_STEPS and target['physical_transitions'] == 1
              and target['native_consumptions'] == 0 and target['old_archive_unchanged'] is True, 'cascade_contract')
    P.require(all(target[k] is False for k in ('integer_continuation_verified', 'full_finalizer_verified',
                                              'quality_gate_clear', 'actual_video_run')), 'cascade_permissions')
    packets = P.lines(output, 'PROBABILISTIC_BASIS.jsonl')
    P.require(len(packets) == 1, 'cascade_single_basis')
    packet, observer = packets[0], P.read(output, 'SETTLED_BASIS_OBSERVER.json')
    initial = serializer.decode(packet['state'])
    P.require(observer['candidate'] == packet['initial_candidate'] and observer['error'] is None
        and observer['restored'] is True and observer['restore_binding_owned'] is True, 'cascade_observer')
    P.require(initial.frame == target['probabilistic_basis_frame'] == observer['candidate']['frame']
              and initial.scope == tuple(observer['candidate']['scope']), 'cascade_initial')
    P.audit_votes(output, packet, canonical, initial)
    FunctionType(P.audit_tracking.__code__, dict(vars(P), POST_UPDATES=V.I.POST_COUNT))(output, packet, initial)
    final = P.read(output, 'BASIS_CASCADE_FINAL.json')
    receipt = final['transition']
    audit_transition(output, initial, receipt)
    following = serializer.decode(receipt['state'])
    P.require(receipt['kind'] == 'basis_cascade' and receipt['next_consumed'] is False
        and receipt['physical_certified'] is False and 0 < receipt['distribution_report']['retained_mass'] <= 1
        and following.scope == initial.scope and following.frame == receipt['applied_frame'] > initial.frame,
        'cascade_transition')
    P.require(final['active_origin'] is False and final['final_state'] == 'stable'
              and final['final_time'] >= final['chain_until'], 'cascade_closed')
    retired = P.read(output, 'BASELINE_RETIREMENT.json')
    P.require(retired['used'] is True and retired['tracker_restored'] is True
        and len([row for row in retired['rows'] if row['suppressed']]) == 1, 'baseline_retired_once')
    P.require(P.read(output, 'INFLIGHT_QUARANTINE.json')['references_restored'] is True, 'quarantine_restore')
    return following, dict(source_call_token=packet['source_call_token'], basis_frame=initial.frame,
        final_frame=following.frame, saved_joint_decoded=True, actual_video=False, GPU_GO=False,
        quality_gate_clear=False, runtime_finalization_allowed=False, integer_continuation_verified=False)


def audit_transition(output: Any, initial: Any, receipt: Any) -> None:
    rows = P.lines(output, 'PROBABILISTIC_TRACKING.jsonl')
    inputs = P.lines(output, 'BASIS_ONLY_INPUT.jsonl')
    expected = list(range(initial.frame + V.I.STRIDE, inputs[-1]['frame'] + V.I.STRIDE, V.I.STRIDE))
    P.require([row['scope']['frame_idx'] for row in rows] == expected, 'cascade_tracking_contiguous')
    found = [row for row in rows if row.get('reason') == 'basis_cascade_applied']
    P.require(len(found) == 1 and found[0]['transition'] == receipt
        and found[0]['journal_token'] == receipt['source_call_token']
        and found[0]['scope']['frame_idx'] == receipt['applied_frame'], 'cascade_saved_transition_link')
    origin = receipt['origin']
    P.require(origin['first_observed_frame'] < receipt['applied_frame'], 'cascade_origin_order')
    journal = [row for row in P.lines(output, 'atomic_journal.jsonl') if row['kind'] == 'step']
    for frame, token in ((origin['first_observed_frame'], origin['source_call_token']),
                         (receipt['applied_frame'], receipt['source_call_token'])):
        selected = [row for row in journal if row['frame_idx'] == frame and row['side'] == initial.scope[-1]]
        P.require(len(selected) == 1, 'cascade_original_J_count')
        row = selected[0]
        P.require(row['token'] == token and row['status'] == 'returned' and row['exception'] is None
            and row['source_id'] == initial.scope[0] and row['run_id'] == initial.scope[1]
            and row['software_reset'] == initial.scope[2] and row['pipe_object_id'] == initial.scope[3],
            'cascade_original_J_scope_token')
    retired = P.read(output, 'BASELINE_RETIREMENT.json')
    P.require(all(row['settlement_token'] == receipt['source_call_token']
                  for row in retired['rows'] if row['suppressed']), 'baseline_settlement_token')


def live(kept: Any) -> Any:
    state, factory = kept['state'], kept['factory']
    connection, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    serializer = type(connection).__init__.__globals__['S']
    decoded, report = saved(state['output'], serializer)
    current, recovery = connection.registry.current(connection.binding), connection.recovery
    anchor = P.read(state['output'], 'PROBABILISTIC_FACTORY_ANCHOR.json')
    P.require(anchor['ready'] is True and anchor['error'] is None and anchor['factory_id'] == id(factory)
        and anchor['registry_id'] == id(connection.registry) and anchor['binding_id'] == id(connection.binding)
        and anchor['source_call_token'] == connection.binding.initial_call_token
        and anchor['frame'] == current.frame and anchor['scope'] == list(current.scope), 'cascade_anchor')
    P.require(connection.registry.factory is factory and state['_g2_probabilistic_scope_registry'] is connection.registry
        and recovery.factory is factory and current == decoded, 'cascade_live_registry')
    P.require(connection.binding.scope == current.scope and connection.rows == 1
        and connection.binding.initial_call_token == report['source_call_token'], 'cascade_binding')
    P.require(recovery.error is None and recovery.failure is None and mode.error is None
        and recovery.reset_count == 1 and recovery.baseline_count == 0 and recovery.pending is None, 'cascade_recovery')
    P.require(mode.native is not None and not mode.native.pending and not mode.native.seen_occurrences
        and len(mode.applied) == 1 and mode.basis_cascade_closed and recovery.control.history.get('1P') is None,
        'cascade_live_transition')
    P.require(recovery.evidence.scope(factory, recovery.pipe) == current.scope
        and recovery.journal.active is None and recovery.journal.steps == V.J_STEPS, 'cascade_live_scope')
    journal = recovery.journal
    P.require(journal.closed and not journal.errors and journal.steps == len(journal.expected)
              and not state.get('baseline_cleanup_errors'), 'cascade_J_complete_cleanup')
    audit_issued(journal, mode.applied[0], current.scope[-1])
    A.retained(state, state['repeat_scope_guard'].reset_lease)
    P.require(len(recovery.archive) == 1, 'cascade_archive_count')
    old, original, record = recovery.archive[0]
    P.require(old.owner.state is original and asdict(original) == record['owner'], 'cascade_archive')
    report.update(live_factory_verified=True, cpu_basis_cascade_finalization_verified=True,
                  normal_next_verified=False, integer_finalizer_called=False)
    return report


def audit_issued(journal: Any, receipt: Any, side: str) -> None:
    for frame, token in ((receipt['applied_frame'], receipt['source_call_token']),
        (receipt['origin']['first_observed_frame'], receipt['origin']['source_call_token'])):
        index = journal.expected.index((frame, side))
        P.require(token == 'step:' + str(index) and index < journal.steps, 'cascade_live_issued_token')


def wrap(previous: Any) -> Any:
    return FunctionType(P.wrap.__code__, dict(vars(P), live=live))(previous)
