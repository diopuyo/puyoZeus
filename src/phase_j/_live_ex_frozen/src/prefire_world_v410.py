"""X→Y→Zの仮説を進める時計。死亡証明へは接続しない。"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
import numpy as np
from src.phase_j._live_ex_frozen.src.board import Board, BOARD_COLS, BOARD_ROWS, DEATH_COL, DEATH_ROW
from src.phase_j._live_ex_frozen.src.exchange_fast_expand import drop_ojama_fast
from src.phase_j._live_ex_frozen.src.prefire_ledger_v410 import Ledger

SIDES = (0, 1)
GENERATE, CONFIRM, FINISH, LOCK, START = range(5)


@dataclass(frozen=True)
class Event:
    t: float
    kind: int
    side: int
    source: str
    amount: int = 0
    board: tuple[int, ...] = ()
    trigger: float | None = None
    before: tuple[int, ...] = ()


def board_of(raw: tuple[int, ...] | list[int]) -> Board:
    return Board.from_list(np.asarray(raw, dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS).tolist())


@dataclass
class World:
    now: float
    boards: list[tuple[int, ...]]
    ledger: Ledger
    events: list[Event] = field(default_factory=list)
    dead: list[bool] = field(default_factory=lambda: [False, False])
    invalid: str | None = None

    def copy(self) -> World:
        return World(self.now, list(self.boards), self.ledger.copy(), list(self.events),
                     list(self.dead), self.invalid)

    def add(self, events: list[Event]) -> None:
        if any(e.t < self.now for e in events):
            raise ValueError('仮説イベントの時刻が過去')
        self.events.extend(events)
        self.events.sort(key=lambda e: (e.t, e.kind, e.side))

    def drop(self, side: int, stamp: float, trigger: float | None = None) -> None:
        count = self.ledger.land(side, stamp, trigger)
        board = board_of(self.boards[side])
        # 端数配置は1通りの代表。これを死亡断定の証拠にはしない。
        landed = drop_ojama_fast(board, count, tuple(range(count % BOARD_COLS)))
        self.boards[side] = tuple(int(c) for c in landed._grid.ravel())
        self.dead[side] |= landed.is_dead()

    def _event(self, event: Event) -> None:
        if event.kind == CONFIRM:
            self.ledger.confirm(event.source, event.t)
        elif event.kind == START:
            if event.before != self.boards[event.side] or self.dead[event.side]:
                self.invalid = 'preparation_changed_by_interception'
        elif event.kind in (FINISH, LOCK):
            if self.dead[event.side]:
                return
            if event.before and self.boards[event.side] != event.before:
                self.invalid = 'preparation_changed_by_interception'
                return
            self.boards[event.side] = event.board
            self.dead[event.side] |= bool(event.board[DEATH_ROW*BOARD_COLS+DEATH_COL])
            self.drop(event.side, event.t, event.trigger if event.kind == FINISH else None)

    def advance(self, until: float) -> None:
        if until < self.now:
            raise ValueError('時計の巻戻し')
        if any(self.dead):
            self.events = []
            return
        due = [e for e in self.events if e.t <= until]
        self.events = [e for e in self.events if e.t > until]
        for stamp, grouped in groupby(due, key=lambda e: e.t):
            events = list(grouped)
            sends = [e for e in events if e.kind == GENERATE and not self.dead[e.side]]
            amounts = tuple(sum(e.amount for e in sends if e.side == i) for i in SIDES)
            sources = tuple(next((e.source for e in sends if e.side == i), '') for i in SIDES)
            if any(amounts):
                self.ledger.generate(amounts, stamp, sources)
            for event in events:
                if event.kind != GENERATE:
                    self._event(event)
            if any(self.dead):
                self.events, self.now = [], stamp
                return
        self.now = until

    def finish(self) -> None:
        self.advance(max([self.now]+[e.t for e in self.events]))

    def audit(self) -> dict:
        return dict(self.ledger.audit(), projected_dead=list(self.dead), invalid=self.invalid,
                    end_sec=self.now, boards=[list(b) for b in self.boards])
