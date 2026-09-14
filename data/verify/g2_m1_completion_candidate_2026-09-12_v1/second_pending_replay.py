"""2P未処理票を原J消費と保存反映/退役票から再構成。物理分布の検収は別。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import native_consumption as N


def require(ok: bool, reason: str) -> None:
    if not ok: raise ValueError('second_pending_replay:' + reason)


def encoded(event: Any) -> dict:
    value = asdict(event)
    value['pair'] = list(value['pair'])  # 保存JSONと同じ表現。RAMのtupleとの誤比較をしない。
    return value


def step_for(steps: dict[int, dict], frame: int, scope: list) -> dict:
    step = steps[frame]
    require(step['frame_idx'] == frame and step['status'] == 'returned' and step['exception'] is None, 'step_success')
    require([step['source_id'], step['run_id'], step['software_reset'], step['pipe_object_id'],
             step['generation']['reset_epoch'], step['side']]
            == [scope[i] for i in (0, 1, 2, 3, 5, 6)], 'step_scope')
    return step


def apply(pending: list[dict], receipt: dict, step: dict) -> None:
    require(receipt['source_call_token'] == step['token'] and receipt['applied_frame'] == step['frame_idx'], 'receipt_call')
    if receipt.get('kind') == 'basis_cascade':
        require(not pending and receipt['next_consumed'] is False, 'basis_cascade_consumption')
        return
    require(bool(pending) and receipt['occurrence_token'] == pending[0]['occurrence_token']
            and receipt['consumed_frame'] == pending[0]['consumed_frame'], 'receipt_head')
    pending.pop(0)


def retirement(packet: dict, step: dict, pending: list[dict], scope: list) -> None:
    following = packet['new_scope']
    require(packet['old_state']['scope'] == scope and all(following[i] == scope[i] for i in (0, 1, 3, 4, 6))
            and all(following[i] >= scope[i] for i in (2, 5))
            and any(following[i] > scope[i] for i in (2, 5)), 'retirement_scope')
    require([step['source_id'], step['run_id'], step['pipe_object_id'], step['side']]
            == [scope[i] for i in (0, 1, 3, 6)]
            and step['software_reset'] == packet['start_epoch']
            and step['generation_after']['reset_epoch'] == following[5], 'retirement_J_scope')
    require(step['status'] == 'returned' and step['exception'] is None
            and step['frame_idx'] == packet['frame'] and step['code_sha256'] == packet['code_sha256'], 'retirement_J')
    require(packet['pending'] == pending and packet['pending_discarded'] is False, 'retirement_pending')
    event = N.extract(dict(scope=dict(frame_idx=packet['frame']), token=step['token'], events=step['events']))
    require(packet['source_call_token'] == step['token'] and packet['reset_call_consumption']
            == (None if event is None else encoded(event)), 'retirement_call')


def timeline(mode: dict, steps: dict[int, dict], end: int) -> dict[int, tuple[str, ...]]:
    initial, retired = mode['initial'], mode['retired']
    scope, start = initial['state']['scope'], initial['state']['frame']
    require(scope[-1] == '2P' and start <= end, 'initial')
    first = step_for(steps, start, scope)
    require(initial['source_call_token'] == first['token'], 'initial_call')
    stop = end + N.FRAME_STRIDE if retired is None else retired['frame']
    require(start < stop <= end + N.FRAME_STRIDE, 'retirement_clock')
    require(all((frame - start) % N.FRAME_STRIDE == 0 for frame in steps if start <= frame <= min(stop, end)), 'extra_off_grid_step')
    receipts: dict[int, dict] = {}
    for receipt in mode['applied']:
        frame = receipt['applied_frame']
        require(start < frame < stop and frame not in receipts, 'receipt_frame')
        receipts[frame] = receipt
    pending: list[dict] = []
    seen: set[str] = set()
    result = {start: ()}
    for frame in range(start + N.FRAME_STRIDE, stop, N.FRAME_STRIDE):
        step = step_for(steps, frame, scope)
        event = N.extract(dict(scope=dict(frame_idx=frame), token=step['token'], events=step['events']))
        if event is not None:
            require(event.occurrence_token not in seen, 'duplicate_consumption')
            seen.add(event.occurrence_token)
            pending.append(encoded(event))
        if frame in receipts: apply(pending, receipts[frame], step)
        result[frame] = tuple(v['occurrence_token'] for v in pending)
    if retired is not None:
        retirement(retired, steps[stop], pending, scope)
    return result
