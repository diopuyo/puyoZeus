"""現在原票に対応する固定rootの参考読取。物理同一性や未来攻撃の認証ではない。"""
from dataclasses import asdict
import json
from typing import Any
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction
import journal_pair_reader as R

SCOPE_KEYS = ('source_id', 'run_id', 'pipe_object_id', 'software_reset')
FPS = 60
CHAIN_STATE = 'CHAIN'


def packet(side: str, frame: int, reason: str, origin: Any = None, instance: int | None = None) -> dict:
    return dict(kind='fixed_origin_reference/v1', side=side, cutoff_frame=frame,
        status='HOLD' if origin is None else 'REFERENCE_ONLY', reason=reason, instance_id=instance,
        origin=None if origin is None else asdict(origin), meaning='fixed_total_not_remaining_attack',
        physical_identity_verified=False, future_fire_power_supply_authorized=False,
        accounting_permission=False, calibrated_win_probability=False, quality_gate_clear=False,
        conditional_board_distribution=None, uncertainty='physical_binding_and_joint_distribution_not_certified')


def snapshot_for(capture: Any, row: dict, side: str) -> Any:
    scope = {key: row[key] for key in SCOPE_KEYS}
    if capture.scopes.get(side) != scope:
        raise ValueError('fixed_origin_reference_unconsumed_scope')
    handle = capture.handles.get(side)
    if handle is None:
        return None
    snapshot = capture.ledger.snapshot(handle)
    current = row['generation_after']
    generation = capture.module.ChainGeneration(side, current['reset_epoch'], current['action_revision'])
    if handle.side != side or snapshot.generation != generation:
        raise ValueError('fixed_origin_reference_generation_or_owner')
    if snapshot.status.value != 'provisional':
        raise ValueError('fixed_origin_reference_retired_handle')
    return snapshot


def qualification(capture: Any, row: dict, snapshot: Any, side: str) -> str | None:
    returned = row.get('returned') or {}
    raw = returned.get('active_origin')
    if raw is None: return 'returned_origin_missing'
    if capture.settled(raw): return 'returned_origin_is_settled_notice'
    if returned.get('state') != CHAIN_STATE: return 'returned_state_not_chain'
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
    if encoded not in capture.seen[side]: return 'returned_origin_not_consumed'
    first = snapshot.episodes[0]
    landings = {(event.trigger_sec, event.before_sha256) for event in snapshot.episodes
                if event.mechanism == 'landing'}
    if landings != {(first.trigger_sec, first.before_sha256)}:
        return 'multiple_landing_contents_in_same_instance'
    return None


def read(capture: Any, side: str, frame: int) -> dict:
    if side not in R.SIDES or type(frame) is not int or frame < 0:
        raise ValueError('fixed_origin_reference_input')
    if capture.closed or capture.error is not None or capture.last_frame != frame:
        raise ValueError('fixed_origin_reference_not_completed_current_update')
    pair = R.read_pair(capture.witness, capture.journal, capture.pipe, frame)
    rows = json.loads(pair.rows_json)
    if not rows: return packet(side, frame, ';'.join(pair.holds))
    row = next(item for item in rows if item['side'] == side)
    if row['generation'] != row['generation_after']:
        return packet(side, frame, 'generation_changed_within_step')
    snapshot = snapshot_for(capture, row, side)
    if snapshot is None: return packet(side, frame, 'fixed_origin_handle_missing')
    reason = qualification(capture, row, snapshot, side)
    if reason: return packet(side, frame, reason, instance=snapshot.handle.instance_id)
    origin = _origin_prediction(snapshot)
    if origin is None: return packet(side, frame, 'fixed_origin_reference_missing')
    if origin.available_at.frame_idx > frame or origin.available_at.time_sec > frame / FPS:
        raise ValueError('fixed_origin_reference_future_input')
    return packet(side, frame, 'software_episode_reference_not_physical_binding', origin, snapshot.handle.instance_id)


def save(capture: Any, frame: int) -> dict:
    """元consumer成功後に同streamへ追記し、失敗時は元原票と例外を保存する。"""
    source = None
    try:
        source = R.read_pair(capture.witness, capture.journal, capture.pipe, frame).rows_json
        result = dict(kind='fixed_origin_reference_pair/v1', frame=frame,
            sides=[read(capture, side, frame) for side in R.SIDES], quality_gate_clear=False)
        capture.stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
        capture.stream.flush()
        return result
    except BaseException as error:
        capture.error = error
        capture.record_failure(frame, source, error)
        raise
