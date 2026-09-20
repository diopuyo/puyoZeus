"""旧公開と確率追跡の保存時間軸を原Jへ照合。確率分布検査は別責務。"""
from __future__ import annotations
import json
from typing import Any

STRIDE = 2
SIDE = '1P'


def indexed(rows: list[dict[str, Any]], field: str) -> dict[int, dict[str, Any]]:
    result = {row[field]: row for row in rows}
    assert len(result) == len(rows), 'duplicate_frame'
    return result


def join_step(row: Any, steps: Any, outer: Any, scope: tuple[Any, ...], *, waiting: bool) -> None:
    frame = row['frame'] if waiting else row['scope']['frame_idx']
    step = steps[frame]
    assert step['token'] == row['journal_token'] and step['status'] == 'returned'
    assert step['exception'] is None and step['software_reset'] == scope[2]
    assert step['source_id'] == scope[0] and step['run_id'] == scope[1]
    assert step['pipe_object_id'] == scope[3] and step['side'] == SIDE
    assert outer[frame]['tickets_this_update'] == outer[frame]['released_this_update'] == 0
    if waiting:
        assert tuple(row['scope']) == scope and row['baseline'] is False and row['reason'] is not None
        assert not any(event['stage'] in ('fifo_before', 'fifo_after') for event in step['events'])
    else:
        observed = row['scope']
        assert (observed['source_id'], observed['run_id'], observed['pipe_object_id'], observed['side']) == (
            scope[0], scope[1], scope[3], SIDE)
        assert observed['generation']['reset_epoch'] == scope[2]
        assert row['integer_current_published'] is False and row['quality_gate_clear'] is False
        # 通常手のnative消費・確率遷移は許容。FIFO無しを追跡区間へ流用しない。


def waiting_rows(history: Any, recovery: Any, proof_frame: int, basis_frame: int) -> list[Any]:
    assert not any(row['kind'] == 'new_baseline' for row in recovery), 'integer_baseline_present'
    waits = [row for row in recovery if row['kind'] == 'reset_baseline_wait']
    recorded = [row for row in history if row.get('kind') == 'reset_baseline_wait']
    assert [row['frame'] for row in waits] == list(range(proof_frame + STRIDE, basis_frame + STRIDE, STRIDE))
    assert waits and json.dumps(waits, sort_keys=True, allow_nan=False) == json.dumps(
        recorded, sort_keys=True, allow_nan=False), 'wait_saved_original'
    return waits


def check(consumer: Any, history: Any, journal: Any, recovery: Any, tracking: Any,
          *, history_first: int, end_frame: int, proof_frame: int, basis_frame: int,
          scope: tuple[Any, ...]) -> dict[str, Any]:
    outer = indexed(consumer, 'frame_idx')
    frames = list(outer)
    assert frames and frames == list(range(frames[0], frames[-1] + STRIDE, STRIDE)), 'consumer_coverage'
    assert all(type(frame) is int for frame in (history_first, end_frame, proof_frame, basis_frame))
    assert frames[0] <= history_first <= proof_frame < basis_frame <= frames[-1] == end_frame
    assert all(row['same_result_identity'] and row['comparison_completed'] and not row['changed_sides']
               and row['full_before'] == row['full_after'] for row in consumer), 'consumer_changed'
    assert scope[-1] == SIDE and proof_frame < basis_frame <= frames[-1]
    owned = [row for row in history if 'decision' in row]
    assert owned and all('decision' in row or row.get('kind') == 'reset_baseline_wait' for row in history)
    own_frames = [row['scope']['frame_idx'] for row in owned]
    # 開始/終端は削除され得る保存票自身から推測せず、呼出元の実owner/最終callで固定する。
    assert own_frames == list(range(history_first, proof_frame + STRIDE, STRIDE)), 'old_history_coverage'
    waits = waiting_rows(history, recovery, proof_frame, basis_frame)
    later = [row['scope']['frame_idx'] for row in tracking]
    assert later == list(range(basis_frame + STRIDE, frames[-1] + STRIDE, STRIDE)), 'tracking_coverage'
    steps = indexed([row for row in journal if row['kind'] == 'step' and row['side'] == SIDE], 'frame_idx')
    for row in waits:
        join_step(row, steps, outer, scope, waiting=True)
    for row in tracking:
        join_step(row, steps, outer, scope, waiting=False)
    issued = [row['scope']['frame_idx'] for row in owned if row['decision']['current_permission']]
    actual = [frame for frame, row in outer.items() for _ in range(row['tickets_this_update'])]
    assert actual == issued and all(frame <= proof_frame for frame in issued), 'issued_frame_join'
    assert sum(row['released_this_update'] for row in consumer) == 0, 'unexpected_release'
    return dict(waiting_updates=len(waits), tracking_updates=len(tracking), issued_frames=actual,
                legacy_publication_join_verified=True, actual_factory=False,
                probabilistic_distribution_verified=False, quality_gate_clear=False)
