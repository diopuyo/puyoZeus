"""v4.9e: 応手列挙の出力不変の高速化と、決定的な作業量による打切り (既定OFF)。

v4.9d の応手列挙 (`prefire_responses_v49d.enumerate_responses`) は、1配置ごとに連鎖の全段をPythonの
オブジェクト (盤面2枚・消去グループの集合) へ組み直し、得点計算では使わない全消し判定まで行っていた。
ここでは同じ結果 (最終盤面・素点) だけを、native の連鎖トレースから直接取り出す。

- 出力不変 (`work_limit=None`): v4.9d の `enumerate_responses` と全く同じ応手集合・得点・順序を返す。
- 作業量の打切り (`work_limit` を指定): 連鎖シミュレーション回数と段数の重み付き和を決定的な作業量とし、
  手数2段目だけを得点の高い順に読み、作業量が上限に達した時点で、完了した盤面までの結果を返す。
  壁時計は使わない。1手目は常に最後まで読む。出力は v4.9d と変わりうる (別版として採点する)。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable
import numpy as np
from src.board import Board, BOARD_COLS, BOARD_ROWS
from src.chain import ChainSimulator
from src.exchange_event_multilanding import PAIR_SIZE, SearchCutoff, EXACT_FAST_GROUPS
from src.exchange_fast_expand import FastExpander, local_pop, new_board
from src.exchange_multilanding_precheck import exceeds_limit, placement_count
from src.exchange_native_chain import BACKEND
from src.prefire_responses_v49d import (RESPONSE_MAX_HANDS, RESPONSE_NODE_LIMIT, representative,
                                        feasible_responses as feasible_responses_v49d)
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import calculate_step_score

# 作業量 (低スペック換算ミリ秒の予測値) = NODE_WEIGHT*ノード数 + SIM_WEIGHT*連鎖シミュレーション回数
# + STEP_WEIGHT*連鎖の総段数。重みは遅い更新の応手列挙の計測時間への非負最小二乗で決めた値 (事前登録に記録)。
# 打切りを使わない (`work_limit=None`) ときは参照しない。
NODE_WEIGHT = 0.0052
SIM_WEIGHT = 0.1100
STEP_WEIGHT = 0.0342
# 応手の1手目は常に読む。打切りの対象にするのは、この番号以降の段 (0始まり)。
FIRST_TRUNCATABLE_HAND = 1
# 応手集合の再利用 (盤面・ツモ・手数・色・上限が同じなら同じ結果)。
RESPONSE_MEMO_LIMIT = 2048
COLORS = (1, 2, 3, 4)


@dataclass(frozen=True)
class _Group:
    """得点計算が読む最小限の消去グループ (色と個数)。"""
    color: int
    size: int


@dataclass(frozen=True)
class _Step:
    """`calculate_step_score` が読む最小限の連鎖1段。"""
    chain_index: int
    erased_groups: tuple
    erased_count: int


class LeanResult:
    """連鎖があった配置の結果。最終盤面・素点・連鎖数だけを持つ。"""
    __slots__ = ('chain_count', 'final_board', 'score')

    def __init__(self, chain_count: int, final_board: Board, score: int) -> None:
        self.chain_count, self.final_board, self.score = chain_count, final_board, score


def lean_chain(board: Board, hidden: bool, native: object) -> LeanResult:
    """`chain_result` + `calculate_chain_score(...).total_score` と同じ最終盤面・素点を、段ごとの盤面を作らずに得る。"""
    trace = native.simulate_chain_trace_py(board._grid.ravel().tolist(), hidden)
    if not trace:
        return LeanResult(0, board.copy(), 0)
    total = 0
    for index, (_before, _after, groups, _ojama, _score) in enumerate(trace, 1):
        erased = tuple(_Group(color, len(cells)) for color, cells, _adjacent in groups)
        total += calculate_step_score(_Step(index, erased, sum(g.size for g in erased))).score
    final = new_board(np.array(trace[-1][1], dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS))
    return LeanResult(len(trace), final, total)


class LeanExpander(FastExpander):
    """`FastExpander` と同じ配置の列挙・並びで、連鎖があった子盤面だけを `lean_chain` で解く。"""

    def __init__(self, simulator: ChainSimulator, native: object, counters: dict) -> None:
        super().__init__(simulator)
        self.native, self.counters, self.cache = native, counters, {}

    def child(self, base: Board, stable: bool, cells: list) -> tuple[Board, LeanResult | None]:
        grid = base._grid.copy()
        for row, col, color in cells:
            grid[row, col] = color
        placed = new_board(grid)
        if stable and not local_pop(grid, [(r, c) for r, c, _ in cells], self.hidden):
            return placed, None
        key = grid.tobytes()
        result = self.cache.get(key)
        if result is None:
            result = self.cache[key] = lean_chain(placed, self.sim._exclude_hidden_row_from_pop, self.native)
            self.counters['sims'] += 1
            self.counters['steps'] += result.chain_count
        return placed, result


class LeanSearch:
    """試合で使われる4色だけで未知ツモを展開する応手列挙器 (v4.9d の `ResponseSearch` と同じ列挙)。"""

    def __init__(self, native: object, limit: int, colors: tuple) -> None:
        self.nodes, self.limit, self.colors = 0, limit, tuple(colors)
        self.counters = dict(sims=0, steps=0)
        self.expander = LeanExpander(ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
                                                    fast_groups=EXACT_FAST_GROUPS), native, self.counters)

    def work(self) -> float:
        return NODE_WEIGHT*self.nodes+SIM_WEIGHT*self.counters['sims']+STEP_WEIGHT*self.counters['steps']

    def pairs_for(self, queue: tuple, hand: int) -> list[tuple]:
        pair = queue[PAIR_SIZE*hand:PAIR_SIZE*(hand+1)]
        known = len(pair) == PAIR_SIZE and all(v in (1, 2, 3, 4, 5) for v in pair)
        return [tuple(pair)] if known else list(combinations_with_replacement(self.colors, PAIR_SIZE))

    def reserve(self, cost: int) -> None:
        """これから数えるノードで上限を超えるなら、探索せず打切りにする (v4.9d と同じ段単位の先読み)。"""
        if exceeds_limit(self.nodes, cost, self.limit):
            self.nodes = self.limit+1
            raise SearchCutoff('node_limit')

    def outcomes(self, current: Board, pairs: list[tuple]) -> list[tuple[Board, int]]:
        """全配置 (1配置 = 1ノード) の (連鎖後盤面, 素点)。"""
        stable, found = self.expander.is_stable(current), []
        for colors in pairs:
            for _, placed, result in self.expander.placements(current, colors, stable):
                self.nodes += 1
                found.append((placed, 0) if result is None else (result.final_board, result.score))
        return found


def enumerate_lean(board: Board, queue: tuple, hands: int, colors: tuple, native: object,
                   limit: int = RESPONSE_NODE_LIMIT, work_limit: float | None = None,
                   max_hands: int = RESPONSE_MAX_HANDS) -> dict:
    """手数 `hands` までの全配置を、連鎖解決後の盤面と累計得点の組で返す (窒息する手は除く)。

    `work_limit=None` なら v4.9d の `enumerate_responses` と同じ集合。指定すると、2手目以降の盤面を
    得点の高い順に読み、作業量が上限に達したら、完了した盤面までの結果で打ち切る (`truncated=True`)。
    """
    search = LeanSearch(native, limit, colors)
    frontier, replies, done, truncated = [(board, 0)], {}, 0, False
    try:
        for hand in range(max(0, min(hands, max_hands))):
            pairs = search.pairs_for(queue, hand)
            search.reserve(len(pairs)*sum(placement_count(b._grid) for b, _ in frontier))
            budgeted = work_limit is not None and hand >= FIRST_TRUNCATABLE_HAND
            if budgeted:
                frontier = sorted(frontier, key=lambda r: (-r[1], int(np.count_nonzero(r[0]._grid)), r[0]._grid.tobytes()))
            level: dict = {}
            for current, points in frontier:
                if budgeted and search.work() >= work_limit:
                    truncated = True
                    break
                for final, score in search.outcomes(current, pairs):
                    if final.is_dead():
                        continue
                    key = final._grid.tobytes()
                    previous = level.get(key, (None, -1))[1]
                    level[key] = (final, max(previous, points+int(score)))
            frontier = list(level.values())
            for key, (final, points) in level.items():
                replies[key] = (final, max(points, replies.get(key, (None, -1))[1]))
            done = hand+1
            if truncated:
                break
    except SearchCutoff:
        return dict(replies=list(replies.values()), complete=False, nodes=search.nodes, hands_done=done,
                    truncated=False, work=search.work(), **search.counters)
    return dict(replies=list(replies.values()), complete=not truncated, nodes=search.nodes, hands_done=done,
                truncated=truncated, work=search.work(), **search.counters)


class Replies:
    """順位づけ済みの応手列 (盤面バイト列・得点)。盤面は参照するときに作る。

    応手集合は1配置ごとに盤面オブジェクトを作ると、再利用表に数十万個の GC 追跡対象が残り、世代2の回収が
    数百ミリ秒止まる (v4.9d で観測した遅い更新の一因)。バイト列と整数だけなら GC に追跡されない。
    """
    __slots__ = ('grids', 'points')

    def __init__(self, grids: list[bytes], points: list[int]) -> None:
        self.grids, self.points = grids, points

    def __len__(self) -> int:
        return len(self.grids)

    @staticmethod
    def board_of(raw: bytes) -> Board:
        return new_board(np.frombuffer(raw, dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS).copy())

    def __getitem__(self, index: int | slice) -> 'tuple[Board, int] | Replies':
        if isinstance(index, slice):
            return Replies(self.grids[index], self.points[index])
        return self.board_of(self.grids[index]), self.points[index]

    def __iter__(self) -> Iterable[tuple[Board, int]]:
        for raw, points in zip(self.grids, self.points):
            yield self.board_of(raw), points


def compact(replies: Iterable[tuple[Board, int]]) -> Replies:
    """決定的な順序 (得点の大きい順、ぷよの少ない順、盤面バイト列順) に並べて、バイト列の列にする。"""
    ordered = representative(replies)
    return Replies([b._grid.tobytes() for b, _ in ordered], [int(p) for _, p in ordered])


_MEMO: dict[tuple, dict] = {}
META_KEYS = ('complete', 'nodes', 'hands_done', 'truncated', 'work', 'sims', 'steps')


def ranked_responses(board: Board, queue: tuple, hands: int, colors: tuple, native: object,
                     limit: int = RESPONSE_NODE_LIMIT, work_limit: float | None = None,
                     max_hands: int = RESPONSE_MAX_HANDS) -> dict:
    """応手列挙 (順位づけ済み・バイト列) の再利用つき入口。v4.9d の `feasible_responses` + `representative` と同じ順序の応手を返す。

    native の連鎖トレースが使えない文脈では v4.9d の列挙へ戻る (出力は同一、再利用は v4.9d 側の表)。
    """
    if BACKEND.get() is None:
        found = feasible_responses_v49d(board, queue, hands, 0., colors, limit)
        return dict(ranked=compact(found['replies']), complete=found['complete'], nodes=found['nodes'])
    key = (board._grid.tobytes(), tuple(queue), int(hands), tuple(colors), int(limit), work_limit, int(max_hands))
    if key not in _MEMO:
        if len(_MEMO) >= RESPONSE_MEMO_LIMIT:
            _MEMO.pop(next(iter(_MEMO)))
        found = enumerate_lean(board, queue, hands, colors, native, limit, work_limit, max_hands)
        _MEMO[key] = dict(ranked=compact(found['replies']), **{k: found[k] for k in META_KEYS})
    return _MEMO[key]
