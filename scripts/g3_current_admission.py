"""保存済みの同frame現在資格を既存採録判定へ追加する。"""
from __future__ import annotations

from typing import Any, Callable
from scripts import g3_live_current_guard as L

C, A = L.C, L.C.A
KEY = 'g3_current_admission'


class Gate:
    """資格の保存完了だけを保持し、原盤面や過去履歴を所有しない。"""

    def __init__(self, state: dict) -> None:
        self.current, self.observer = state[C.KEY], state[A.KEY]
        self.pending: dict | None = None
        self.latest: dict | None = None
        self.busy = False
        self.error: str | None = None

    def capture(self, original: Callable, observer: Any, *args: Any) -> dict:
        """最終live guardの出力を同consume内で一回だけ捕捉する。"""
        row = original(observer, *args)
        if observer is self.observer:
            A.require(self.busy and self.pending is None, 'current_admission_capture_order')
            A.require(row.get('live_context_checked') is True, 'current_admission_live_guard_missing')
            self.pending = row
        return row

    def consume(self, original: Callable, current: Any, *args: Any) -> None:
        """元writer成功と行数増加の後にだけ資格を昇格する。"""
        if current is not self.current:
            return original(current, *args)
        A.require(not self.busy and self.error is None, 'current_admission_failed_or_reentry')
        self.busy, self.pending, self.latest = True, None, None
        before = current.rows
        try:
            original(current, *args)
            A.require(self.pending is not None and current.rows == before + 1
                      and current.save_error is None, 'current_admission_unsaved')
            self.latest = dict(row=self.pending, rows=current.rows, frames=self.observer.frames)
        except BaseException as error:
            self.error = repr(error)
            raise
        finally:
            self.busy, self.pending = False, None

    def permitted(self, frame: int, clock: float) -> bool:
        """未保存、前frame、握られた例外を採録許可へ変えない。"""
        current, saved = self.current, self.latest
        if self.busy or self.error or current.error or current.save_error or current.stream.closed or saved is None:
            return False
        row = saved['row']
        return (saved['rows'] == current.rows and saved['frames'] == self.observer.frames
                and (row['frame'], row['time_sec']) == (frame, clock)
                and row['status'] == 'CURRENT_INPUT_ELIGIBLE' and row['live_context_checked'] is True)

    def decide(self, original: Callable, observer: Any, frame: int, clock: float,
               side: str, eligible: bool) -> bool:
        """旧UI条件とoriginal_eligibleを維持し、保存済資格だけ追加する。"""
        if observer is not self.observer:
            return original(observer, frame, clock, side, eligible)
        row = observer.latest
        A.require(observer.error is None and observer.active is None and row is not None, 'admission_before_update')
        A.require(row['frame'] == frame and row['time_sec'] == clock and side in A.SIDES, 'admission_frame')
        qualified = self.permitted(frame, clock)
        allowed = bool(eligible and row['status'] == A.OBSERVED and qualified)
        observer.decisions += 1
        observer.withheld += int(eligible and not allowed)
        try:
            observer.write(dict(kind='admission', frame=frame, time_sec=clock, side=side,
                original_eligible=bool(eligible), allowed=allowed, ui_status=row['status'],
                current_saved_eligible=qualified, current_hold_reason=None if qualified else 'CURRENT_NOT_SAVED_ELIGIBLE',
                quality_gate_clear=False))
        except BaseException as error:
            self.error = observer.save_error = repr(error)
            self.latest = None
            raise
        return allowed


def install(stack: Any, state: dict, replace: Callable) -> Gate:
    """既存の3参照を同stackで差替・復元する。原_should_emitは触らない。"""
    A.require(KEY not in state and L.KEY in state and C.KEY in state, 'current_admission_install_order')
    A.require(state[C.KEY].rows == 0 and state[A.KEY].decisions == 0, 'current_admission_late')
    gate = state[KEY] = Gate(state)
    selection, consume, decide = C.selection, C.CurrentExit.consume, A.Admission.decide
    replace(stack, C, 'selection', lambda observer, *args: gate.capture(selection, observer, *args))
    replace(stack, C.CurrentExit, 'consume', lambda current, *args: gate.consume(consume, current, *args))
    replace(stack, A.Admission, 'decide', lambda observer, *args: gate.decide(decide, observer, *args))
    return gate
