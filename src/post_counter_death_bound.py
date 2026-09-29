"""打ち返し後の全応手を、楽観火力上限と窒息の必要条件で縮約する。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
import math

import numpy as np

from src.board import Board, BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN
from src.chain import ChainSimulator, MIN_ERASE_COUNT
from src.post_counter_geometry import BoundSimulator, usable_colors, all_landings_dead, place_pair
from src.scoring import OJAMA_MAX_DROP_PER_TURN, ALL_CLEAR_BONUS, compute_effective_rate
from src.scoring import BASE_SCORE_PER_PUYO, MAX_BONUS_MULTIPLIER, chain_power, color_bonus, connection_bonus

PAIR_SIZE = 2
GAME_COLORS = 4
COLORS = (1, 2, 3, 4, 5)
ROTATIONS = 4
NODE_LIMIT = 2048  # 計算予算。超過時は必ず従来探索へ戻す。
TURN_LIMIT = BOARD_ROWS
DROP_SCORE_UPPER = BOARD_ROWS * PAIR_SIZE


@lru_cache(maxsize=BOARD_ROWS * BOARD_COLS)
def step_upper(count: int, colors: int = GAME_COLORS) -> int:
    """同時消しの全整数分割を緩和DPで覆い、連結・多色ボーナスを落とさない。"""
    bonus = [0] * (count + 1)
    for total in range(MIN_ERASE_COUNT, count + 1):
        bonus[total] = max(connection_bonus(group) + bonus[total-group]
                           for group in range(MIN_ERASE_COUNT, total + 1))
    return bonus[count] + color_bonus(min(colors, count // MIN_ERASE_COUNT))


@lru_cache(maxsize=BOARD_ROWS * BOARD_COLS)
def score_upper(count: int) -> int:
    """消せる総数から、段数と各段の消去配分を全て覆う得点上限を返す。"""
    @lru_cache(maxsize=None)
    def best(left: int, stage: int) -> int:
        if left < MIN_ERASE_COUNT:
            return 0
        return max(n * BASE_SCORE_PER_PUYO * max(1, min(MAX_BONUS_MULTIPLIER,
            chain_power(stage) + step_upper(n))) + best(left-n, stage+1)
            for n in range(MIN_ERASE_COUNT, left+1))
    return best(count, 1)


class Undecided(Exception):
    """証明不能を死亡と混同しない。"""


@dataclass
class Bound:
    """楽観相殺と全端数配置を共有する、一回限りの証明状態。"""
    queue: tuple[int, ...]
    palette: tuple[int, ...]
    rate: int
    limit: int
    simulator: ChainSimulator = field(default_factory=lambda: BoundSimulator(exclude_hidden_row_from_pop=True))
    nodes: int = 0
    pruned: int = 0
    cache: dict = field(default_factory=dict)
    bounds: list[dict] = field(default_factory=list)

    def pairs(self, turn: int) -> list[tuple[int, int]]:
        """既知NEXT/NEXT2以降は試合4色の全組合せを認める。"""
        pair = self.queue[PAIR_SIZE*turn:PAIR_SIZE*(turn+1)]
        if len(pair) == PAIR_SIZE and all(c in self.palette for c in pair):
            return [pair]
        return list(combinations_with_replacement(self.palette, PAIR_SIZE))

    def visit(self, board: Board, pending: int, turn: int, grace: int) -> bool:
        """上限でも相殺できない枝を続行し、打切り・生存は断定を拒否する。"""
        if pending <= 0:
            return False
        if board.is_dead():
            return True
        key = (board._grid.tobytes(), pending, min(turn, PAIR_SIZE), grace)
        if key in self.cache:
            return self.cache[key]
        self.nodes += 1
        if self.nodes > self.limit or turn >= TURN_LIMIT:
            raise Undecided('bound_limit')
        pairs = self.pairs(turn)
        supply = Counter({c: max(pair.count(c) for pair in pairs) for c in self.palette})
        usable = usable_colors(board, supply)
        # 非連鎖手にも落下点と端数の即時相殺を無償付与し、死ににくくする。
        sent = math.ceil(DROP_SCORE_UPPER / self.rate)
        left = max(0, pending-sent)
        if left == 0:
            return False
        if not usable and grace == 0 and all_landings_dead(board, min(left, OJAMA_MAX_DROP_PER_TURN)):
            self.pruned += 1
            self.bounds.append(dict(turn=turn, usable=usable, score_upper=0, incoming=pending,
                                    cancel_upper=sent, drop=min(left, OJAMA_MAX_DROP_PER_TURN)))
            self.cache[key] = True
            return True
        result = self.expand(board, pending, turn, grace, pairs)
        self.cache[key] = result
        return result

    def land(self, board: Board, pending: int) -> list[Board]:
        """端数を受け側に最も有利に選べるよう、全配置を保持する。"""
        drop = min(pending, OJAMA_MAX_DROP_PER_TURN)
        return [self.simulator.drop_ojama_with_remainder_columns(board, drop, cols)
                for cols in combinations(range(BOARD_COLS), drop % BOARD_COLS)]

    def expand(self, board: Board, pending: int, turn: int, grace: int,
               pairs: list[tuple[int, int]]) -> bool:
        """連鎖手は着弾を延期し、非連鎖手だけ30個ずつ降らせる。"""
        seen: set[tuple] = set()
        simulate = getattr(self.simulator, 'simulate_reply', self.simulator.simulate)
        for pair in pairs:
            for rotation in range(ROTATIONS):
                for col in range(BOARD_COLS if rotation % PAIR_SIZE == 0 else BOARD_COLS-1):
                    placed = place_pair(board, pair, col, rotation)
                    if placed is None:
                        continue
                    result = simulate(placed)
                    identity = (result.final_board._grid.tobytes(), bool(result.chain_count))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    # 実際に使った色数まで上限を絞り、同じ色ぷよの火力を重複付与しない。
                    power = score_upper(result.total_erased)
                    # 全消し持越しの有無を入力で確定できないため、各発火へ無償付与する。
                    bonus = ALL_CLEAR_BONUS if result.chain_count else 0
                    sent = math.ceil((power + bonus + DROP_SCORE_UPPER) / self.rate)
                    left = max(0, pending-sent)
                    if left == 0 and not result.final_board.is_dead():
                        return False
                    postpone = result.chain_count > 0 or grace > 0
                    boards = [result.final_board] if postpone else self.land(result.final_board, left)
                    remaining = left if postpone else left-min(left, OJAMA_MAX_DROP_PER_TURN)
                    if any(not b.is_dead() and not self.visit(b, remaining, turn+1, max(0, grace-1))
                           for b in boards):
                        return False
        return True


def prove_post_counter(board: Board, queue: tuple[int, ...], incoming: int, hands: int,
                       elapsed: float, palette: tuple[int, ...], node_limit: int = NODE_LIMIT) -> dict:
    """不明セル・浮遊・色未確認は証明しない。死亡済み盤面も新規断定しない。"""
    grid = board._grid
    if (grid.shape != (BOARD_ROWS, BOARD_COLS) or np.any(grid == COLOR_UNKNOWN)
            or np.any((grid[:-1] != 0) & (grid[1:] == 0)) or board.is_dead()
            or hands < 1 or incoming <= 0 or len(set(palette)) != GAME_COLORS
            or not set(palette) <= set(COLORS)
            or not set(int(c) for c in np.unique(grid) if c in COLORS) <= set(palette)
            or any(c not in (0, COLOR_UNKNOWN, *palette) for c in queue)):
        return dict(dead=False, reason='invalid_bound_input', nodes=0, pruned=0, bounds=[])
    proof = Bound(queue, palette, compute_effective_rate(elapsed), node_limit)
    try:
        dead = proof.visit(board, incoming, 0, hands-1)
        reason = 'post_counter_upper_bound' if dead else 'optimistic_survivor'
    except Undecided as exc:
        dead, reason = False, str(exc)
    counts = Counter(tuple(row.items()) for row in proof.bounds)
    bounds = [dict(key, branches=count) for key, count in counts.items()]
    return dict(dead=dead, reason=reason, nodes=proof.nodes, pruned=proof.pruned, bounds=bounds)
