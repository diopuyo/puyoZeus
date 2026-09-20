"""原update復帰後のG3現在評価資格。原結果・物理・内部診断票は変更しない。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from scripts import g3_admission as A
from scripts.g3_agent_review import encoded, save

KEY = 'g3_current_exit'
SIDES = (('1P', 'p1'), ('2P', 'p2'))
ROWS, COLUMNS = 13, 6


def selection(observer: Any, pipe: Any, result: Any, frame: int, clock: float) -> dict:
    """当該UIと両STABLEを結合する。採録済・人手真値・予測許可ではない。"""
    row = observer.latest
    A.require(observer.pipe is pipe and observer.active is None and row is not None, 'current_owner')
    A.require(observer.error is None and observer.save_error is None, 'current_ui_failed')
    A.require((row['frame'], row['time_sec']) == (frame, clock)
              == (result.frame_idx, result.time_sec), 'current_clock')
    reasons, boards = [], {}
    if row['status'] != A.OBSERVED:
        reasons.append(A.UNKNOWN)
    if not result.is_match_active or result.match_end_locked:
        reasons.append('MATCH_INACTIVE_OR_LOCKED')
    for side, name in SIDES:
        value = getattr(result, name)
        if value.state.value != 'stable' or value.confirmed_board is None:
            reasons.append(side + '_NOT_CONFIRMED_STABLE')
            continue
        grid = value.confirmed_board._grid
        A.require(grid.shape == (ROWS, COLUMNS), 'current_board_shape')
        # 隠し段を観測真値へ昇格させず、現在評価の可視入力だけを保存する。
        boards[side] = grid[1:].tolist()
    return dict(kind='g3_current_exit', frame=frame, time_sec=clock,
                status='HOLD' if reasons else 'CURRENT_INPUT_ELIGIBLE', reasons=reasons,
                visible_confirmed_boards=None if reasons else boards,
                hidden_row_observed=False, human_gt=False, collector_append_verified=False,
                projected_prediction_permitted=False, model_evaluation_completed=False,
                quality_gate_clear=False)


class CurrentExit:
    """資格原票だけを所有し、表示凍結やmodel推論の状態を新設しない。"""

    def __init__(self, observer: Any, cls: Any) -> None:
        self.observer, self.cls = observer, cls
        self.original = vars(cls)['update']
        self.output = Path(observer.output)
        self.stream = (self.output / 'G3_CURRENT_EXIT.jsonl').open('xb')
        self.rows = self.eligible = 0
        self.error: str | None = None
        self.save_error: str | None = None

    def consume(self, pipe: Any, result: Any, frame: int, clock: float) -> None:
        """元update成功後のみ保存。write失敗を正常資格へ変えない。"""
        row = selection(self.observer, pipe, result, frame, clock)
        row.update(source_id=self.observer.source_id, run_id=self.observer.run_id)
        try:
            self.stream.write(encoded(row) + b'\n')
            self.stream.flush()
        except BaseException as error:
            self.save_error = repr(error)
            raise
        self.rows += 1
        self.eligible += int(row['status'] == 'CURRENT_INPUT_ELIGIBLE')

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        """元例外を優先し、閉鎖/復元/保存不備を区別する。"""
        failure: BaseException | None = None
        if body is not None:
            self.error = self.error or repr(body)
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException as error:
            failure, self.save_error = error, repr(error)
        try:
            self.stream.close()
        except BaseException as error:
            failure, self.save_error = failure or error, repr(error)
        restored = vars(self.cls).get('update') is self.original
        try:
            save(self.output / 'G3_CURRENT_EXIT_STATUS.json', dict(rows=self.rows, eligible=self.eligible,
                 error=self.error, save_error=self.save_error, closed=self.stream.closed,
                 references_restored=restored, quality_gate_clear=False))
        except BaseException as error:
            failure = failure or error
        if body is None:
            if failure is not None:
                raise failure
            A.require(restored, 'current_reference_restore')
        return False


def install(stack: Any, collector: Any, state: dict, replace: Callable[..., Any]) -> CurrentExit:
    """Admission設置直後の同じ内側stackへ一度だけ接続する。"""
    A.require(KEY not in state and A.KEY in state, 'current_install_order_or_duplicate')
    observer = state[A.KEY]
    A.require(observer.frames == 0 and observer.active is None, 'current_install_late')
    value = CurrentExit(observer, collector.RecognitionPipeline)
    state[KEY] = value
    stack.push(value.close)
    original = value.original
    def update(pipe: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        A.require(value.error is None and value.save_error is None and not value.stream.closed,
                  'current_exit_failed')
        try:
            result = original(pipe, frame_idx, time_sec, frame)
            value.consume(pipe, result, frame_idx, time_sec)
            return result
        except BaseException as error:
            value.error = repr(error)
            raise
    replace(stack, collector.RecognitionPipeline, 'update', update)
    return value
