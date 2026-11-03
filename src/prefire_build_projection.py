"""明示呼出専用の伸ばし予測。既存本番経路からは参照しない。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import heapq
from typing import Any
import numpy as np
from src.b3_timing import TimingTable, column_heights
from src.exchange_count_ledger import gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC
from src.prefire_build_ledger import BuildLedger
from src.prefire_worker import DEATH_CELL

VISIBLE, DEPTH, MAX_HANDS = 3, 8, 128
COLS, ROWS = 6, 13
GENERATE, FINISH, PLACE = 0, 1, 2
EPS = 1e-9


def drop_one(board: list, col: int, color: int) -> bool:
    """native::drop_one同様、空洞へ詰めず最上ぷよの上へ置く。"""
    top = next((r for r in range(ROWS) if board[r*COLS+col]), ROWS)
    if top == 0:
        return False
    board[(top-1)*COLS+col] = color
    return True


def place(board: list, pair: tuple, x: int, rotation: int) -> list:
    out = list(board)
    a, b = pair if rotation != 2 else pair[::-1]
    other = x+1 if rotation == 1 else x-1 if rotation == 3 else x
    assert drop_one(out,x,a) and drop_one(out,other,b), 'ama合法手の再現失敗'
    return out


@dataclass
class Decisions:
    """更新間の再利用は可視情報だけで鍵化する。"""
    values: dict = field(default_factory=dict)
    hits: int = 0

    def first(self, native: Any, board: list, visible: list, colors: list,
              width: int, pending: int, reuse: bool) -> tuple | None:
        key = (tuple(board), tuple(map(tuple, visible)), tuple(colors), width, DEPTH, pending)
        if reuse and key in self.values:
            self.hits += 1
            return self.values[key]
        # k=1ではama_fがNEXTを見ないため、k=depthとし可視3組のみを渡す。
        value = native.ama_f_py(board, list(map(tuple, visible)), DEPTH, width, DEPTH, colors,
            incoming=pending, stop=True, budget_ms=None)
        first = tuple(value['steps'][0]) if value['steps'] else None
        if reuse:
            self.values[key] = first
        return first


class Projection:
    def __init__(self, request: dict, tapes: list, carries: list, native: Any,
                 width: int, delta: float, cache: Decisions, reuse: bool,
                 core: Any = None) -> None:
        self.boards = [list(b) for b in request['boards']]
        self.tapes, self.colors = tapes, request['colors']
        self.cursors, self.reasons, self.carries = [0, 0], [None, None], list(carries)
        self.native, self.width, self.delta = native, width, delta
        self.cache, self.reuse, self.elapsed = cache, reuse, request['elapsed_sec']
        self.table = TimingTable()
        self.ledger = BuildLedger(request['pending'], core, request['input_sec'])
        self.events = []
        self.shots, self.moves = [], []

    def schedule(self, side: int, stamp: float) -> None:
        if self.reasons[side] == '着弾溢れ':
            return
        if self.boards[side][DEATH_CELL]:
            self.reasons[side] = '窒息'
            return
        cursor = self.cursors[side]
        visible = self.tapes[side][cursor:cursor+VISIBLE]
        if len(visible) < VISIBLE or cursor >= MAX_HANDS:
            self.reasons[side] = 'ツモ欠測・手数上限'
            return
        move = self.cache.first(self.native, self.boards[side], visible, self.colors,
            self.width, self.ledger.pending(side), self.reuse)
        if move is None:
            self.reasons[side] = '合法手なし'
            return
        _, x, rotation, _, _, _ = move
        heights = column_heights(np.asarray(self.boards[side]).reshape(ROWS, COLS))
        other = x+1 if rotation == 1 else x-1 if rotation == 3 else x
        seconds = self.table.hand_sec(float(max(heights)), float(max(heights[x], heights[other])))[0]
        if stamp+seconds <= self.delta+EPS:
            heapq.heappush(self.events, (stamp+seconds, PLACE, side, move))

    def placed(self, side: int, stamp: float, move: tuple) -> None:
        _, x, rotation, score, chain, board = move
        pair = self.tapes[side][self.cursors[side]]
        raw = place(self.boards[side], pair, x, rotation)
        trace = self.native.simulate_chain_trace_py(raw, True)
        assert sum(t[4] for t in trace) == score
        self.boards[side] = list(board)
        self.cursors[side] += 1
        self.moves.append((side, stamp, x, rotation, pair, score))
        if not chain:
            self.drop(side, stamp)
            self.schedule(side, stamp)
            return
        end = stamp+max(chain*self.table.stage_sec+self.table.tail_sec,
                       FORMULA_SCORE_DELAY_SEC+(chain-1)*STEP_SEC)
        key = (side, len(self.shots))
        self.shots.append(dict(side=side, start=stamp, end=end, scores=[t[4] for t in trace]))
        for stage, step in enumerate(trace, 1):
            time = stamp+FORMULA_SCORE_DELAY_SEC+(stage-1)*STEP_SEC
            heapq.heappush(self.events, (time, GENERATE, side, (step[4], key, end)))
        heapq.heappush(self.events, (end, FINISH, side, None))

    def drop(self, side: int, stamp: float) -> None:
        count = self.ledger.drop(side, stamp)
        for i in range(count):
            if not drop_one(self.boards[side],i % COLS,9):
                self.reasons[side] = '着弾溢れ'

    def generate(self, side: int, stamp: float, data: tuple, ledger: BuildLedger,
                 carries: list) -> None:
        score, key, end = data
        amount, carries[side] = gross([score], [self.elapsed+stamp], carries[side], 0.)
        ledger.send(side, amount, key, end)

    def run(self) -> dict:
        for side in (0, 1):
            self.schedule(side, 0.)
        while self.events and self.events[0][0] <= self.delta+EPS:
            stamp, kind, side, data = heapq.heappop(self.events)
            if kind == GENERATE:
                self.generate(side, stamp, data, self.ledger, self.carries)
            elif kind == FINISH:
                self.drop(side, stamp)
                self.schedule(side, stamp)
            else:
                self.placed(side, stamp, data)
        # 連鎖途中の終了予測盤と未反映送りを対にする既存v4.5方式。
        projected, carries = deepcopy(self.ledger), list(self.carries)
        for stamp, kind, side, data in sorted(self.events):
            if kind == GENERATE:
                self.generate(side, stamp, data, projected, carries)
        return dict(boards=self.boards, pending=[projected.pending(s) for s in (0, 1)],
            cursors=self.cursors, reasons=self.reasons, moves=self.moves, shots=self.shots,
            landed=self.ledger.landed, pending_at_delta=[self.ledger.pending(s) for s in (0, 1)])


def project(request: dict, tapes: list, carries: list, native: Any, width: int,
            delta: float, cache: Decisions | None = None, enabled: bool = False,
            core: Any = None) -> dict | None:
    if not enabled:
        return None
    if set(request) - {'game', 'generation', 'input_sec', 'evidence_times', 'boards', 'known',
                        'colors', 'pending', 'elapsed_sec', 'stable', 'seed'}:
        raise ValueError('未来情報を含む未許可入力欄')
    if max(request['evidence_times']) > request['input_sec'] or not all(request['stable']):
        raise ValueError('未来証跡または非STABLE')
    return Projection(request, tapes, carries, native, width, delta,
                      cache if cache is not None else Decisions(), cache is not None, core).run()
