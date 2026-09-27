"""F1bと同じ発火前観測から非飽和火力・打ち返し余地を作る。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from src import indicators_v2 as iv
from src import puyo_core_bridge as native
from src.board import Board, BOARD_COLS, BOARD_ROWS
from src.chain import ChainSimulator
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import calculate_chain_score, compute_effective_rate, score_to_ojama

RESPONSE_WIDTH = 2 * BOARD_COLS + 2 * (BOARD_COLS - 1)
CACHE_SIZE = 32768
NEW_COLUMNS = tuple(f"NF_ojama_k{k}_{side}" for side in ("self", "opp", "diff")
                    for k in iv.NEAR_FUTURE_K_LEVELS) + ("counter_margin_self", "counter_margin_opp")


@dataclass(frozen=True)
class CountObservation:
    """発火前盤面・NEXTを参照共有せず保持する。"""

    grids: np.ndarray
    queues: np.ndarray
    elapsed_sec: float

    def __post_init__(self) -> None:
        ivalue = (("grids", (2, BOARD_ROWS, BOARD_COLS)), ("queues", (2, 4)))
        for name, shape in ivalue:
            value = np.array(getattr(self, name), dtype=np.int8, copy=True)
            if value.shape != shape:
                raise ValueError(f"{name}の形状が不正")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class NativeSimulator:
    """学習と同じnative連鎖計算を局所的に選択する。"""

    def simulate(self, board: Board) -> Any:
        """採用済みの隠し段規則で厳密得点を返す。"""
        return native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


@lru_cache(maxsize=CACHE_SIZE)
def fire(raw: bytes, queue: tuple[int, ...], elapsed: float,
         levels: tuple[int, ...], response: bool = False) -> np.ndarray:
    """学習時の探索幅・既知NEXT・換算率で生個数を計算する。"""
    grid = np.frombuffer(raw, dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    board = Board.from_list(grid.tolist())
    width = RESPONSE_WIDTH if response else iv.NEAR_FUTURE_BEAM_WIDTH
    result = iv.near_future_fire_power(board, queue[:2], queue[2:], elapsed,
        simulator=NativeSimulator(), k_levels=levels, beam_width=width,
        active_colors=iv._near_future_active_colors(board), resolve_before_death=response,
        use_exact_score=True)
    return np.asarray([result.values[k].raw for k in levels])


@lru_cache(maxsize=CACHE_SIZE)
def completion(raw: bytes, firing: bool, elapsed: float) -> tuple[float, int, bytes]:
    """既発火側のみ完走させ、応手で同じ火力を再加算しない。"""
    board = Board.from_list(np.frombuffer(raw, np.int8).reshape(BOARD_ROWS, BOARD_COLS).tolist())
    if not firing:
        return 0., 0, raw
    result = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).simulate(board)
    sent = score_to_ojama(calculate_chain_score(result).total_score, elapsed_sec=elapsed).ojama_count
    return sent, result.chain_count, result.final_board._grid.astype(np.int8).tobytes()


def side_features(observation: CountObservation, firing: tuple[bool, bool],
                  before: np.ndarray | None = None, after: np.ndarray | None = None) -> np.ndarray:
    """S1は完走予測、S3は得点差とn=1。どちらも同じ発火前観測を使う。"""
    from src.exchange_event_landing import remaining_hands

    elapsed = observation.elapsed_sec
    completed = [completion(grid.tobytes(), active, elapsed)
                 for grid, active in zip(observation.grids, firing)]
    sends = np.array([row[0] for row in completed])
    counts = [row[1] for row in completed]
    if before is not None:
        missing = ~np.isfinite(before) | ~np.isfinite(after) | (before < 0) | (after < 0)
        sends = np.where(missing, np.nan, np.maximum(after-before, 0)/compute_effective_rate(elapsed))
        counts = [0, 0]
    values = np.empty((2, len(iv.NEAR_FUTURE_K_LEVELS) + 1))
    for side, grid in enumerate(observation.grids):
        queue = tuple(int(v) if 1 <= v <= 5 else 0 for v in observation.queues[side])
        values[side, :-1] = np.log1p(fire(grid.tobytes(), queue, elapsed, iv.NEAR_FUTURE_K_LEVELS))
        busy = iv.estimate_chain_anim_duration_sec(counts[side], "empirical_table_2026_08_14")
        hands = remaining_hands(counts[1-side], 0., 0., busy_sec=busy)
        level = hands - iv.NEAR_FUTURE_KNOWN_HAND_SLOTS
        available = fire(completed[side][2], queue, elapsed, (level,), True)[0]
        margin = available - np.maximum(sends[1-side] - sends[side], 0)
        values[side, -1] = np.sign(margin) * np.log1p(abs(margin))
    return values


def orient(values: np.ndarray, side: int) -> np.ndarray:
    """絶対量は側交換のみ、差分だけ反転する。"""
    own, opp = values[[side, 1-side]]
    return np.r_[own[:-1], opp[:-1], own[:-1]-opp[:-1], own[-1], opp[-1]]
