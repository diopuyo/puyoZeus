"""原journal/NEXT/生rawを同callerへ束縛し、履歴消費を実FIFOで検査する。"""
from __future__ import annotations

from collections import Counter
from typing import Any
import transaction as T
import history_dependencies as D
import history_state as H

FIRST, STRIDE, FPS, SIDE = 34796, 2, 60, '1P'
RAW_MISSING = object()


class Provider(D.V.Provider):
    """時刻上限で所有を放棄しない。未記録scopeやresetは元viewで拒否する。"""

    def selected(self, pipe: Any, side: str, frame: int, clock: float) -> bool:
        return side == SIDE and type(frame) is int and frame >= FIRST

    def baseline_selected(self, view: Any) -> bool:
        return view.frame == FIRST and view.scope[-1] == SIDE

    def adjacent(self, previous: tuple[int, float], view: Any) -> bool:
        return previous[0] + STRIDE == view.frame and view.clock == view.frame / FPS

    def raw(self, pipe: Any, side: str, view: Any) -> tuple[Any, dict[str, Any]]:
        history = self.journal.history
        memory = getattr(pipe, '_stable_color_memory_' + side.lower())
        H.require(history.frame == view.frame and history.time_sec == view.clock, 'history_raw_clock')
        entry = history.capture.get(id(memory))
        H.require(type(entry) is dict and type(entry['captured_frame']) is int
                  and entry['captured_frame'] == view.frame, 'history_raw_not_captured')
        raw = entry.get('raw')
        H.require(type(raw) is dict and type(raw.get('grid')) is list, 'history_raw_grid_missing')
        grid = tuple(tuple(row) for row in raw['grid'])
        # 元Recorderはcopy済み値を同frameに保持。live identityは元view側で検査する。
        return grid, {'captured_frame': view.frame, 'raw_grid': grid, 'capture': entry}

    def no_origin(self, pipe: Any, side: str, view: Any) -> bool:
        origin = getattr(pipe, '_active_chain_' + side.lower(), RAW_MISSING)
        return origin is None

    def before_history_consume(self, call: Any, caller: Any) -> None:
        view, values, rec = call['view'], caller.f_locals, self.journal
        H.require(rec.active is not None and rec.active['frame'] is caller
                  and rec.active['scope']['frame_idx'] == view.frame, 'history_actual_step')
        old = self.owner(values['self'], values['side'], view.scope[2])
        H.require(old['queue'] is view.queue and tuple(old['tokens']) == view.tokens
                  and len(old['refs']) == len(view.refs)
                  and all(a is b for a, b in zip(old['refs'], view.refs)), 'history_prepared_FIFO_changed')
        invocation = rec.controller._invocation(values['self'], view.frame, view.clock)
        bound, main, slide = self.invocations[id(caller)]
        H.require(invocation is bound and invocation.main is main and invocation.slides[values['side']] is slide
                  and invocation.error is None, 'history_prepared_NEXT_changed')
        H.require(T.board_key(values['prev_confirmed']) == call['binding'].current
                  and T.board_key(values['ctx'].confirmed_board) == call['binding'].current,
                  'history_current_mutated_before_pop')
        H.require(self.no_origin(values['self'], values['side'], view), 'history_origin_before_pop')

    def after_history_consume(self, call: Any, caller: Any) -> None:
        rec, view = self.journal, call['view']
        H.require(rec.active is not None and rec.active['frame'] is caller, 'history_consume_caller')
        events = [row for row in rec.active['events'] if row['stage'] == 'fifo_after']
        H.require(len(events) == 1 and events[0]['enqueue_occurrence_token'] == view.tokens[0],
                  'history_actual_journal_token')
        H.require(not rec.errors, 'history_journal_failed')

    def after_history_step(self, call: Any, caller: Any) -> None:
        view, values, binding = call['view'], caller.f_locals, call['binding']
        old = self.owner(values['self'], values['side'], view.scope[2])
        expected = view.tokens[1:] if call['consumed'] else view.tokens
        H.require(tuple(old['tokens']) == expected, 'history_final_FIFO_mismatch')
        H.require(T.board_key(values['ctx'].confirmed_board) == binding.current,
                  'history_downstream_changed_current')
        H.require(not self.journal.errors, 'history_journal_failed')
