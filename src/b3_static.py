"""B3静止指標の値を保った計算経路。探索の深さ・候補・正規化は変更しない。"""
from __future__ import annotations

from inspect import signature
from typing import Callable

from src import indicators_v2 as iv, puyo_core_bridge as core
from src.b3_column_spec import IND_NAMES, RAW_COUNT_INDICATORS
from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN, Board
from src.chain import ChainSimulator

SIMULATOR_NAMES = frozenset(n for n in IND_NAMES if n not in RAW_COUNT_INDICATORS
                            and 'simulator' in signature(getattr(iv, n)).parameters)
DROPS = tuple((col, color) for color in iv.IGNITION_TRIAL_COLORS for col in range(BOARD_COLS))


def min_ignite(board: Board, sim: ChainSimulator) -> iv.IndicatorV2Value:
    """従来と同じ未解決盤面への1個/2個追加をRustの既存バッチ関数で調べる。"""
    grid = board._grid
    # Rustは消去した位置だけを詰めるため、元から穴がある盤面は従来シミュレータを使う。
    holes = ((grid[:-1] != COLOR_EMPTY) & (grid[1:] == COLOR_EMPTY)).any()
    if not core.NATIVE_AVAILABLE or (grid == COLOR_UNKNOWN).any() or holes:
        return iv.min_puyos_to_ignite(board, simulator=sim)
    base = sim.simulate(board).chain_count
    ghost = sim._exclude_hidden_row_from_pop
    metrics = core.chain_metrics_after_drops(board, list(DROPS), ghost)
    n = iv.IGNITION_TRIAL_LIMIT + 1
    if any(m is not None and m[0] > base for m in metrics):
        n = 1
    elif iv.IGNITION_TRIAL_LIMIT >= 2:
        # 1個目の連鎖解決を挟まない。窒息判定も従来の中間盤面にだけ適用する。
        boards = [b for col, color in DROPS
                  if (b := iv._drop_one_color(board, col, color)) is not None and not b.is_dead()]
        maxima = core.max_chain_after_drops_for_boards(boards, list(DROPS), ghost) if boards else []
        if any(value > base for value in maxima):
            n = 2
    score = 1.0 - (float(n) / float(iv.IGNITION_MAX_PUYOS))
    return iv.IndicatorV2Value(score=iv._clamp01(score), raw=float(n))


def evaluate(board: Board, funcs: dict[str, Callable]) -> dict[str, float]:
    """同一盤面内のシミュレーションを共有し、結果だけを外部LRUに渡す。"""
    sim = ChainSimulator(exclude_hidden_row_from_pop=iv.GHOST_CHAIN_RULE_ENABLED, fast_groups=True)
    vals = {}
    for name, func in funcs.items():
        if name == 'min_puyos_to_ignite':
            result = min_ignite(board, sim)
        elif name in SIMULATOR_NAMES:
            result = func(board, simulator=sim)
        else:
            result = func(board)
        vals[name] = float(result.score)
    conn, _ = iv.connectivity_observation(board, simulator=sim)
    vals.update(conn_pair_count=float(conn.pair_count), conn_triple_count=float(conn.triple_count),
                conn_max_group_size=float(conn.max_group_size))
    return vals
