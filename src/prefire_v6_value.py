"""Phase 6の比較専用線形物差し。遷移と候補集合は5Cから変更しない。"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np

from src import prefire_v5_search as base
from src import prefire_v5b_search as exact
from src.prefire_v5c_search import TransitionTable, side_options
from src.indicators_v2 import SEC_PER_HAND
from src.board import BOARD_COLS, BOARD_ROWS, DEATH_COL

BOARD_CACHE_SIZE = 65536
CAPACITY = BOARD_ROWS * BOARD_COLS
DEATH_WEIGHT = 8.0
PENDING_WEIGHT = 2.0
SEND_WEIGHT = 1.0
BOARD_WEIGHT = 1.0
PROBABILITY_EPSILON = 1e-7


def logit(probability: float) -> float:
    """本番値と軽量値の比較を有限のlogitに揃える。"""
    p = float(np.clip(probability, PROBABILITY_EPSILON, 1-PROBABILITY_EPSILON))
    return float(np.log(p/(1-p)))


@lru_cache(maxsize=BOARD_CACHE_SIZE)
def board_value(raw: bytes) -> float:
    """空き容量・同色隣接・中央列余裕の3特徴を0〜1で等重み平均する。"""
    grid = np.frombuffer(raw, dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS)
    colored = (grid >= min(base.sim.PLAYABLE_COLORS)) & (grid <= max(base.sim.PLAYABLE_COLORS))
    adjacent = np.count_nonzero((grid[:, 1:] == grid[:, :-1]) & colored[:, 1:])
    adjacent += np.count_nonzero((grid[1:] == grid[:-1]) & colored[1:])
    max_edges = BOARD_ROWS*(BOARD_COLS-1) + (BOARD_ROWS-1)*BOARD_COLS
    features = (1-np.count_nonzero(grid)/CAPACITY, adjacent/max_edges,
                1-np.count_nonzero(grid[:, DEATH_COL])/BOARD_ROWS)
    return float(np.mean(features))


def light_value(exchange: base.Exchange, elapsed: float) -> float:
    """送量・相殺後残量・窒息・残し盤面だけで候補の大小を比較する。"""
    scores = []
    for side, position in enumerate(exchange.sides):
        sent = max(0, base.sim.send_ojama(position.score, elapsed)-exchange.cancelled[side])
        remaining = position.pending + exchange.dropped[side]
        scores.append(SEND_WEIGHT*sent/(CAPACITY+sent)
                      - PENDING_WEIGHT*remaining/(CAPACITY+remaining)
                      - DEATH_WEIGHT*exchange.dead[side]
                      + BOARD_WEIGHT*board_value(position.board))
    return scores[0]-scores[1]


def select(states: tuple, attacker: int, elapsed: float, table: TransitionTable,
           depth: int | None = None, waiting: bool = False) -> tuple | None:
    """候補を落とさず5Bのminimaxで比較。待機は無発火終端から同じ規則で選ぶ。"""
    def options(position: base.Position, limit: int | None, side: int) -> tuple:
        candidates = side_options(position, limit, side)
        return tuple(p for p in candidates if not p.chains) if waiting and side == attacker else candidates
    return exact.choose(states, attacker, elapsed, light_value, depth=depth,
                        unknown_rollouts=exact.UNKNOWN_ROLLOUTS,
                        resolver=table.resolve, options_fn=options)


def evaluate_selected(choice: tuple, attacker: int, elapsed: float,
                      table: TransitionTable, evaluator: Callable) -> tuple:
    """軽量比較で決まった組合せだけを本番評価器へ渡す。"""
    _, attack, response = choice
    time = elapsed + max(0, attack.consumed-1)*SEC_PER_HAND
    pair = (attack, response) if attacker == 0 else (response, attack)
    return (evaluator(table.resolve(*pair, attacker, time), time), attack, response)


def choose(states: tuple, attacker: int, elapsed: float, table: TransitionTable,
           evaluator: Callable, depth: int | None = None) -> tuple:
    """最善候補と待機を本番評価し、攻撃側に有利な方を選ぶ。食い違いも返す。"""
    best = select(states, attacker, elapsed, table, depth)
    wait = select(states, attacker, elapsed, table, depth, waiting=True)
    candidates = [c for c in (best, wait) if c is not None]
    if not candidates:
        return None, 0.0
    unique = list(dict.fromkeys(candidates))
    values = [selected_value(states, c, attacker, elapsed, table, evaluator, depth) for c in unique]
    index = (max if attacker == 0 else min)(range(len(values)), key=lambda i: values[i][0])
    return values[index], abs(unique[index][0]-logit(values[index][0]))


def selected_value(states: tuple, choice: tuple, attacker: int, elapsed: float,
                   table: TransitionTable, evaluator: Callable, depth: int | None) -> tuple:
    """未知3組目は5Cの8標本を維持し、各標本の選択済み応手だけを本番評価する。"""
    sampled = exact.response_states(states, 1-attacker) if depth is None else (states[1-attacker],)
    if len(sampled) == 1:
        return evaluate_selected(choice, attacker, elapsed, table, evaluator)
    values = []
    for state in sampled:
        positions = list(states)
        positions[1-attacker] = state
        def options(position: base.Position, limit: int | None, side: int) -> tuple:
            return (choice[1],) if side == attacker else side_options(position, limit, side)
        selected = exact.choose(tuple(positions), attacker, elapsed, light_value,
                                resolver=table.resolve, options_fn=options)
        values.append(evaluate_selected(selected, attacker, elapsed, table, evaluator))
    return (sum(v[0] for v in values)/len(values), choice[1], choice[2])
