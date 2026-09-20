"""元ownerの固定originを現在clockで読む。現在盤面・会計・終了を変更しない。"""
from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any
from scripts.chain_end_epoch_shadow_v1 import _origin_prediction

SIDES = ('1P', '2P')
TERMINAL_REASONS = frozenset(('already_closed_once', 'end_epoch_invalidated',
                              'ledger_not_provisional', 'software_generation_mismatch'))


@dataclass(frozen=True)
class OriginReferenceRead:
    status: str
    reason: str
    side: str
    cutoff_frame: int
    cutoff_sec: float
    origin_json: str | None = None
    instance_id: int | None = None
    physical_identity_verified: bool = field(default=False, init=False)
    accounting_permission: bool = field(default=False, init=False)
    live_hook_verified: bool = field(default=False, init=False)


def checked_snapshot(rec: Any, observer: Any, side: str) -> tuple[Any, str | None]:
    if observer.rec is not rec or type(rec.ledger) is not rec.ledger_module.ChainPredictionLedger:
        raise ValueError('origin_reader_owner_mismatch')
    handle = rec.active_handles.get(side)
    if handle is None:
        return None, 'origin_handle_missing'
    snapshot = rec.ledger.snapshot(handle)  # 別ledgerのhandleは元の厳格例外を保持する。
    if handle.side != side:
        raise ValueError('origin_reader_side_mismatch')
    key = (side, handle.instance_id)
    if key in observer._closed:
        return snapshot, 'already_closed_once'
    if key in observer._invalidated:
        return snapshot, 'end_epoch_invalidated'
    if snapshot.status.value != 'provisional':
        return snapshot, 'ledger_not_provisional'
    if snapshot.generation != rec._current_generation(side):
        return snapshot, 'software_generation_mismatch'
    return snapshot, None


def read_origin(rec: Any, observer: Any, side: str, frame: int, time_sec: float,
                expected_instance_id: int | None = None) -> OriginReferenceRead:
    if side not in SIDES or type(frame) is not int or frame < 0 or type(time_sec) not in (int, float) \
            or not math.isfinite(time_sec) or time_sec < 0:
        raise ValueError('origin_reader_input')
    if expected_instance_id is not None and (type(expected_instance_id) is not int or expected_instance_id < 1):
        raise ValueError('origin_reader_expected_instance')
    instance_id = expected_instance_id
    def result(status: str, reason: str, payload: str | None = None) -> OriginReferenceRead:
        return OriginReferenceRead(status, reason, side, frame, float(time_sec), payload, instance_id)
    if not rec._clock_active() or rec.frame != frame or rec.time_sec != time_sec:
        return result('HOLD', 'outside_exact_update_clock')
    snapshot, reason = checked_snapshot(rec, observer, side)
    if expected_instance_id is not None and (snapshot is None or snapshot.handle.instance_id != expected_instance_id):
        return result('TERMINAL_HOLD', 'origin_instance_no_longer_active')
    instance_id = None if snapshot is None else snapshot.handle.instance_id
    if reason:
        return result('TERMINAL_HOLD' if reason in TERMINAL_REASONS else 'HOLD', reason)
    origin = _origin_prediction(snapshot)
    if origin is None:
        if snapshot.origin_prediction_revision is not None:
            raise ValueError('origin_reader_invalid_fixed_reference')
        terminal = bool(snapshot.predictions) or frame > snapshot.created_at.frame_idx \
            or time_sec > snapshot.created_at.time_sec
        return result('TERMINAL_HOLD' if terminal else 'HOLD',
                      'origin_reference_unavailable' if terminal else 'origin_prediction_pending')
    if origin.available_at.frame_idx > frame or origin.available_at.time_sec > time_sec:
        raise ValueError('origin_reader_future_reference')
    payload = dict(instance_id=snapshot.handle.instance_id, generation=asdict(snapshot.generation),
        origin=asdict(origin), meaning='total_from_fixed_origin_not_remaining_attack',
        later_prediction_substitution_allowed=False)
    return result('READY', 'fixed_origin_reference', json.dumps(payload, sort_keys=True, allow_nan=False))
