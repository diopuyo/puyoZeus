"""保存入力専用の競争実験部品。本番に配線せず時間で探索を切らない。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np
from src.b3_timing import TimingTable
from src.exchange_projection import hand_secs, hands_for
from src.exchange_count_ledger import gross
from src.prefire_worker import DEATH_CELL, COLS

HORIZON = 48
DROP = 30
SAMPLES = 4
TAPE_LENGTH = 3 * HORIZON + 3
OLD_PREFIX = 10
EXTENSION_SEED = 20261016


def tapes(seed: int, colors: tuple, known: tuple) -> list[list[tuple[int, int]]]:
    """prefire11の各本の既存prefixを保ち、延長用乱数列を別に固定する。"""
    rng = np.random.default_rng(seed)
    result = []
    for sample in range(SAMPLES):
        first = rng.choice(colors, size=(OLD_PREFIX-len(known), 2))
        extension = np.random.default_rng(np.random.SeedSequence([seed, sample, EXTENSION_SEED]))
        tail = extension.choice(colors, size=(TAPE_LENGTH-OLD_PREFIX, 2))
        result.append(list(known)+[tuple(map(int, p)) for p in np.concatenate((first, tail))])
    return result


def land(board: list[int], pending: int) -> tuple[list[int], int]:
    """ama_f::settleの既存端数列順。1回最大30個。"""
    out = list(board)
    count = min(DROP, max(0, pending))
    for i in range(count):
        col = i % COLS
        empty = [row for row in range(len(out)//COLS) if out[row*COLS+col] == 0]
        if empty:
            out[max(empty)*COLS+col] = 9
    return out, pending-count


def place(board: list[int], pair: tuple, x: int, rotation: int) -> list[int]:
    """保存された合法配置を再生し、段得点をnativeトレースで確認するための盤面。"""
    out = list(board)
    a, b = pair if rotation != 2 else pair[::-1]
    y = x+1 if rotation == 1 else x-1 if rotation == 3 else x
    for col, color in ((x, a), (y, b)):
        row = max(r for r in range(len(out)//COLS) if out[r*COLS+col] == 0)
        out[row*COLS+col] = int(color)
    return out


@dataclass
class Reader:
    """キャッシュは1更新・1試行内だけ。サンプル間や更新間で計測費用を隠さない。"""
    native: Any
    colors: tuple
    tape: list[tuple]
    width: int
    depth: int
    elapsed: float
    table: TimingTable = field(default_factory=TimingTable)
    cache: dict = field(default_factory=dict)

    def curve(self, board: list[int], pending: int, cursor: int, horizon: int,
              width: int | None = None, depth: int | None = None) -> tuple | None:
        if horizon > HORIZON or cursor+horizon+2 > len(self.tape):
            return None
        key = (tuple(board), pending, cursor, horizon, width or self.width, depth or self.depth)
        if key not in self.cache:
            self.cache[key] = self.native.ama_race_detail_py(board, self.tape[cursor:], horizon,
                list(self.colors), width or self.width, depth or self.depth, pending=pending)
        return self.cache[key]

    def duration(self, chain: int) -> float:
        return chain*self.table.stage_sec+self.table.tail_sec if chain else 0.

    def best(self, curve: tuple | None, start: int, end: int, begin: float,
             hand: float, carry: int, origin: int = 0, boundary: bool = False) -> dict | None:
        if curve is None:
            return None
        candidates = []
        for h, chain, scores, board, pending, source in curve[0]:
            if not start <= h <= end or source < origin:
                continue
            offset = max(0, h-origin-1)*hand if boundary else h*hand
            stamp = self.elapsed+begin+offset
            times = [stamp+(i+1)*self.table.stage_sec for i in range(len(scores))]
            amount, remainder = gross(scores, times, carry, 0.)
            candidates.append(dict(h=h, chain=chain, gross=amount, carry=remainder, scores=scores,
                sent=max(0, amount-pending), pending=max(0, pending-amount),
                before_pending=pending, board=board, duration=self.duration(chain),
                start=begin+offset, end=begin+offset+self.duration(chain)))
        if candidates:
            return max(candidates, key=lambda c: (c['sent'], -c['chain'], tuple(c['board']), -c['h']))
        states = [s for s in curve[1] if s[0] <= end]
        if not states:
            return None
        stopped, board, pending = states[-1]
        offset = max(0, stopped-origin-1)*hand if boundary else stopped*hand
        return dict(h=stopped, chain=0, gross=0, carry=carry, sent=0, pending=pending, scores=[],
                    before_pending=pending, board=board, duration=0.,
                    start=begin+offset, end=begin+offset)


def hand(board: list[int], table: TimingTable) -> float:
    return hand_secs(table, np.asarray(board).reshape(-1, COLS))[0]



def second_window(x_end: float, y_start: float, y_end: float, step: float) -> tuple[float, int]:
    """連鎖中に操作ツモは無い。残時間0以下は追加配置を認めない。"""
    begin = max(x_end, y_start)
    remaining = max(0., y_end-begin)
    return begin, hands_for(remaining, step) if remaining > 0 else 0
