"""確率基準only CPUの終了検収。整数/実動画/G2全体の承認とは分離する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any

UPDATES, J_STEPS, POST_UPDATES = 105, 210, 25
STRIDE, ACQUISITION_WAIT = 2, 14
EXPECTED_FINISH = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/finish.py'


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('probabilistic_cpu_finish:' + reason)


def read(output: Path, name: str) -> Any:
    return json.loads((output / name).read_bytes())


def lines(output: Path, name: str) -> list[Any]:
    return [json.loads(line) for line in (output / name).read_text(encoding='utf-8').splitlines()]


def saved(output: Path, serializer: Any) -> tuple[Any, dict[str, Any]]:
    result, target = read(output, 'RESULT.json'), read(output, 'PROBABILISTIC_TARGET_RESULT.json')
    require(result['exit_code'] == 0 and result['updates'] == UPDATES and result['J_steps'] == J_STEPS
            and result['guards_unchanged'] is True and result['gpu'] is False, 'body_result')
    canonical = result['reset_continuation']
    require(all(key in canonical and canonical[key] == value for key, value in target.items()), 'target_subset')
    require(canonical['basis_only'] is True and canonical['actual_J_steps'] == J_STEPS
            and canonical['old_archive_unchanged'] is True and canonical['native_consumptions'] == 0
            and canonical['physical_transitions'] == 0, 'basis_only_contract')
    require(canonical['integer_continuation_verified'] is False and canonical['full_finalizer_verified'] is False
            and canonical['quality_gate_clear'] is False and canonical['actual_video_run'] is False, 'permissions')
    packets = lines(output, 'PROBABILISTIC_BASIS.jsonl')
    require(len(packets) == 1, 'single_basis')
    packet = packets[0]
    value = serializer.decode(packet['state'])  # 保存されたjoint支持を元decoderで実際に検査する。
    observer = read(output, 'SETTLED_BASIS_OBSERVER.json')
    require(observer['candidate'] == packet['initial_candidate'] and observer['error'] is None
            and observer['restored'] is True and observer['restore_binding_owned'] is True, 'observer_basis')
    require(value.frame == canonical['probabilistic_basis_frame'] == observer['candidate']['frame']
            and tuple(observer['candidate']['scope']) == value.scope, 'basis_scope_frame')
    audit_votes(output, packet, canonical, value)
    audit_tracking(output, packet, value)
    require(read(output, 'INFLIGHT_QUARANTINE.json')['references_restored'] is True, 'quarantine_restore')
    return value, dict(saved_evidence_verified=True, saved_joint_decoded=True, live_factory_verified=False,
        basis_frame=value.frame, hidden_worlds=len(value.worlds), source_call_token=packet['source_call_token'],
        scope=list(value.scope), integer_continuation_verified=False, runtime_finalization_allowed=False,
        quality_gate_clear=False, GPU_GO=False)


def audit_votes(output: Path, packet: Any, canonical: Any, value: Any) -> None:
    evidence = read(output, 'QUALIFIED_PRIOR_VOTES.json')
    require(evidence['error'] is None and len(evidence['votes']) == canonical['qualified_votes'] > 0, 'nonempty_votes')
    qualified = {row['frame']: row for row in evidence['qualifications'] if row['eligible'] is True}
    for vote in evidence['votes']:
        frame = vote['frame']
        require(vote['exit_used_as_vote'] is False and vote['frames'] == [frame - 6, frame - 4, frame - 2], 'past_votes')
        for source in [*vote['frames'], frame]:
            require(source in qualified and qualified[source]['scope'] == list(value.scope)
                    and qualified[source]['same_call'] is True, 'vote_same_scope')
    require(qualified[value.frame]['token'] == packet['source_call_token'], 'basis_source_call')


def audit_tracking(output: Path, packet: Any, value: Any) -> None:
    status = read(output, 'PROBABILISTIC_TRACKING_STATUS.json')
    activation = status['activation']
    require(status['error'] is None and status['pending_native_consumptions'] == 0
            and activation['source_call_token'] == packet['source_call_token']
            and activation['frame'] == value.frame, 'activation')
    rows = lines(output, 'PROBABILISTIC_TRACKING.jsonl')
    require(len(rows) == status['rows'] and rows, 'tracking_rows')
    require(all(row['native_consumption'] is None and row['pending_occurrences'] == []
                and row['integer_current_published'] is False for row in rows), 'unexpected_consumption')
    start = activation['acquisition_deadline'] - ACQUISITION_WAIT
    inputs = lines(output, 'BASIS_ONLY_INPUT.jsonl')
    expected = list(range(start, start + POST_UPDATES * STRIDE, STRIDE))
    require([row['frame'] for row in inputs] == expected and all(row['side'] == '1P'
        and row['effective_candidate'] is None and row['invocation_preserved'] is True for row in inputs), 'input_frames')


def live(kept: Any) -> dict[str, Any]:
    state, factory = kept['state'], kept['factory']
    connection, mode = state['probabilistic_basis_connection'], state['probabilistic_tracking_mode']
    serializer = type(connection).__init__.__globals__['S']
    decoded, report = saved(state['output'], serializer)
    current = connection.registry.current(connection.binding)
    recovery = connection.recovery
    require(connection.registry.factory is factory and state['_g2_probabilistic_scope_registry'] is connection.registry
            and recovery.factory is factory and current == decoded, 'live_registry')
    require(connection.binding.scope == current.scope
            and connection.binding.initial_call_token == report['source_call_token'] and connection.rows == 1, 'live_binding')
    require(recovery.error is None and recovery.failure is None and mode.error is None
            and recovery.reset_count == 1 and recovery.baseline_count == 0 and recovery.pending is None, 'live_recovery')
    require(mode.native is not None and not mode.native.pending and not mode.native.seen_occurrences
            and not mode.applied and recovery.control.history.get('1P') is None, 'live_integer_or_hand')
    require(recovery.evidence.scope(factory, recovery.pipe) == current.scope
            and recovery.journal.active is None and recovery.journal.steps == J_STEPS, 'live_scope')
    lease = state['repeat_scope_guard'].reset_lease
    lease.archive.verify()
    require(len(recovery.archive) == 1, 'live_archive_count')
    old, original, record = recovery.archive[0]
    require(old.owner.state is original and asdict(original) == record['owner'], 'live_archive')
    report.update(live_factory_verified=True, cpu_basis_only_finalization_verified=True,
                  integer_finalizer_called=False, actual_video=False)
    return report


def wrap(previous: Any) -> Any:
    def loader(original: Any) -> Any:
        base = previous(original)
        def load(alias: str, path: Any, stack: Any) -> Any:
            module = base(alias, path, stack)
            if alias != '_empty_fused_finish':
                return module
            require(Path(module.__file__).resolve() == EXPECTED_FINISH, 'finish_source')
            def finish(kept: Any) -> None:
                if 'probabilistic_tracking_mode' not in kept['state']:
                    return module.finish(kept)
                report = live(kept)
                kept['runtime'].write(kept['state']['output'] / 'PROBABILISTIC_CPU_FINISH.json', report)
            return N(**(vars(module) | dict(finish=finish)))
        return load
    return loader
