"""既存認識監査の有効期間だけ、ガードの入出力と状態遷移を記録する。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
from typing import Any, Iterator
from weakref import WeakKeyDictionary

from src.board import Board
from src.row1_guard import GUARD_COL, ROW_HIDDEN, ROW_TOP, Row1Guard

CURRENT: ContextVar[Any] = ContextVar('row1_action_audit', default=None)
ACTION_SUFFIX = '.row1_guard.jsonl'


@contextmanager
def action_scope(audit: Any, frame: Any) -> Iterator[None]:
    """フレーム番号を引き渡す。無効時も既存の認識経路は変えない。"""
    path = getattr(audit, 'path', None)
    if path is None:
        yield
        return
    if not hasattr(audit, '_row1_actions'):
        audit._row1_actions = ActionLog(path.with_suffix(ACTION_SUFFIX))
    token = CURRENT.set((audit._row1_actions, frame.index))
    try:
        yield
    finally:
        CURRENT.reset(token)


class ActionLog:
    """判定状態に書き込まず、作動した分岐を事後観測する。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.serial = 0
        self.holds: WeakKeyDictionary[Row1Guard, str] = WeakKeyDictionary()
        self.shadows: WeakKeyDictionary[Any, dict[str, Row1Guard]] = WeakKeyDictionary()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')

    def shadow(self, pipe: Any, side: str) -> Row1Guard:
        """実ガードの状態・カウンタとは独立した影の履歴を返す。"""
        guards = self.shadows.setdefault(pipe, {})
        return guards.setdefault(side, Row1Guard())

    def apply(self, guard: Row1Guard, board: Board | None, raw: Board | None,
              time_sec: float, in_chain: bool, frame: int, side: str,
              shadow: bool, next_moved: bool = False,
              ojama_falling: bool = False, settled: bool = False) -> Board | None:
        before = dict(guard.counts)
        held = guard._held_since is not None
        previous = guard._prev_cells
        if board is None:
            guard.reset()
            out = None
        else:
            out = guard.step(board, raw, time_sec, in_chain, next_moved, ojama_falling, settled)
        kind, reason = self._action(guard, before, held, board, out)
        if kind is not None:
            values = {f'row{r}_col2': int(board.get(r, GUARD_COL)) if board is not None
                      else (previous or {}).get((r, GUARD_COL))
                      for r in (ROW_HIDDEN, ROW_TOP)}
            evidence = dict(
                signals=dict(formula=in_chain, next_moved=next_moved,
                             ojama_falling=ojama_falling, settled=settled),
                raw_top=int(raw.get(ROW_TOP, GUARD_COL)) if raw is not None else None,
                after_top=int(out.get(ROW_TOP, GUARD_COL)) if out is not None else None,
                held_since=guard._held_since, suppressed=guard._suppressed,
                restored=guard.counts['screen_restored'] > before['screen_restored'],
                next_signal_sec=guard._next_signal_sec,
            )
            self._write(guard, kind, reason, values, frame, time_sec, side, shadow, evidence)
        return out

    def _action(self, guard: Row1Guard, before: dict[str, int], held: bool,
                board: Board | None, out: Board | None) -> tuple[str | None, str | None]:
        for counter, kind, reason in (
            ('spawn_removed', 'released', 'raw_disappeared'),
            ('floating', 'floating', None),
            ('screen_empty', 'screen_empty', 'empty_evidence'),
            ('spawn_held', 'spawn_held', None),
            ('spawn_released_chain', 'released', 'chain_started'),
            ('spawn_released_next', 'released', 'next_moved'),
            ('spawn_released_ojama', 'released', 'ojama_falling'),
            ('screen_restored', 'released', guard.release_reason),
        ):
            if guard.counts[counter] > before[counter]:
                return kind, reason
        if held and guard._held_since is None:
            return 'released', 'no_board' if board is None else 'top_empty'
        if out is not board:
            return 'screen_empty', 'suppression_continues'
        return None, None

    def _write(self, guard: Row1Guard, kind: str, reason: str | None,
               values: dict, frame: int, time_sec: float, side: str, shadow: bool,
               evidence: dict | None = None) -> None:
        hold_start = False
        restored = bool(evidence and evidence.get('restored'))
        if kind == 'spawn_held' or (not restored and reason in (
            'chain_started', 'next_moved', 'ojama_falling',
        )):
            if guard not in self.holds:
                self.serial += 1
                self.holds[guard] = f'{side}-{self.serial}'
                hold_start = True
        hold_id = self.holds.get(guard)
        row = dict(frame=frame, t_sec=time_sec, side=side, kind=kind,
                   before=values, hold_id=hold_id, hold_start=hold_start,
                   release_reason=reason if kind == 'released' else None,
                   reason=reason, shadow=shadow)
        row.update(version=2, guard_revision='2.1', evidence=evidence)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if kind == 'released' and hold_id is not None and guard._held_since is None:
            self.holds.pop(guard, None)
