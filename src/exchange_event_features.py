"""撃ち合い評価の純粋な特徴変換。列順はT09/T11の検証資産と対応する。"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.board import Board, BOARD_ROWS, BOARD_COLS, COLOR_OJAMA
from src import indicators_v2 as iv
from src.scoring import compute_effective_rate

Array = NDArray[np.float64]
CELLS = BOARD_ROWS * BOARD_COLS
PHASE_BOUNDS = (1 / 3, 2 / 3)
PHASE_NAMES = ("early", "middle", "late")
COLOR_MIN, COLOR_MAX = 1, 5
QUEUE_SIZE = 4
D_COLUMNS = (
    "board_color_puyo_total", "board_puyo_total", "center_bulge_color",
    "center_bulge_ojama", "board_ojama_count", "color_diversity_evenness",
    "buried_hole_count", "current_max_chain", "dig_resistance", "ukeyasusa",
    "sub_chain_count", "chain_efficiency", "min_puyos_to_ignite",
    "second_chain_potential", "main_linked_pair_count", "isolated_pair_count",
    "main_linked_ratio", "ignition_point_count", "multi_color_ignition",
    "simultaneous_pop_richness", "chain_articulation_point_count",
    "saturation_chain_upper", "conn_triple_count", "all_clear_bonus_pending",
    "ojama_net_balance", "ojama_forecast", "ojama_net_balance_synced",
    "ojama_margin", "diff_max_column_height", "diff_column_bumpiness",
    "diff_death_margin", "diff_death_margin_neighbor", "diff_conn_pair_count",
    "diff_conn_max_group_size", "diff_board_color_puyo_total",
    "diff_board_puyo_total", "diff_board_ojama_count", "diff_center_bulge_color",
    "diff_center_bulge_ojama", "diff_current_max_chain", "diff_dig_resistance",
    "diff_ukeyasusa", "diff_sub_chain_count", "opp_all_clear_bonus_pending",
    "color_ojama_ratio_own", "color_diff_x_ojama_diff",
)
SIDE_COLUMNS = ("garbage", "color", "conn_pair", "conn_triple", "conn_max",
                "isolated_pair", "k1", "k2", "k3", "k4", "k5")
ARRIVAL_COLUMNS = tuple(f"{side}_{name}" for side in ("own", "opp", "diff")
                        for name in SIDE_COLUMNS) + ("own_first_trigger", "opp_first_trigger")
SCORE_COLUMNS = ("own_sent", "opp_sent", "net_own_minus_opp", "own_more",
                 "opp_more", "own_score_missing", "opp_score_missing")
PHASE_COLUMNS = tuple(f"{kind}_{phase}" for kind in ("fill", "elapsed")
                      for phase in PHASE_NAMES)
G_COLUMNS = D_COLUMNS + ("m0_logit",) + PHASE_COLUMNS + tuple(
    "m0_logit_x_" + name for name in PHASE_COLUMNS)
S1_COLUMNS = D_COLUMNS + ARRIVAL_COLUMNS
S3_COLUMNS = S1_COLUMNS + SCORE_COLUMNS


def validate_elapsed(elapsed_sec: float) -> None:
    """時刻の欠測や逆行を入力境界で拒否する。"""
    if not np.isfinite(elapsed_sec) or elapsed_sec < 0:
        raise ValueError("elapsed_secは有限の非負値が必要")


def fill_phase(own_total: float, diff_total: float) -> int:
    """両側平均の充填率を三分位にする。欠測はT09同様に0補完する。"""
    progress = own_total - diff_total / 2
    progress = np.clip(progress, 0, 1) if np.isfinite(progress) else 0.0
    return int(np.searchsorted(PHASE_BOUNDS, progress, side="left"))


def g_features(design: Array, probability_1p: Array, sign: Array,
               phases: NDArray[np.integer]) -> Array:
    """自側D・1P確率・視点符号から59列を作る。位相は反転しない。"""
    epsilon = np.finfo(np.float64).eps
    p = np.clip(probability_1p, epsilon, 1 - epsilon)
    logits = ((np.log(p) - np.log1p(-p)) * sign)[:, None]
    dummy = np.eye(len(PHASE_NAMES))[phases.astype(int)]
    fill, elapsed = dummy[:, 0], dummy[:, 1]
    return np.column_stack((design, logits, fill, elapsed, logits * fill, logits * elapsed))


def prefire_side_features(grid: NDArray, queue: NDArray, elapsed_sec: float) -> Array:
    """確定盤面と既知NEXTから形・おじゃま・K=1..5火力を算出する。"""
    validate_elapsed(elapsed_sec)
    grid, queue = np.asarray(grid), np.asarray(queue)
    if grid.shape != (BOARD_ROWS, BOARD_COLS) or queue.shape != (QUEUE_SIZE,):
        raise ValueError("盤面は13×6、NEXTは4色が必要")
    if not np.isin(grid, (0, 1, 2, 3, 4, 5, 9, 10)).all():
        raise ValueError("盤面の色コードが不正")
    board = Board.from_list(grid.astype(int).tolist())
    conn, _ = iv.connectivity_observation(board)
    colors = (grid >= COLOR_MIN) & (grid <= COLOR_MAX)
    shape = [np.count_nonzero(grid == COLOR_OJAMA) / CELLS,
             np.count_nonzero(colors) / CELLS, conn.pair_count / (CELLS / 2),
             conn.triple_count / (CELLS / 3), conn.max_group_size / CELLS,
             iv.isolated_pair_count(board).score]
    q = np.where((queue >= COLOR_MIN) & (queue <= COLOR_MAX), queue, 0).astype(int)
    fire = np.zeros(len(iv.NEAR_FUTURE_K_LEVELS))
    if not board.is_dead():
        result = iv.near_future_fire_power(
            board, tuple(q[:2]), tuple(q[2:]), elapsed_sec,
            active_colors=iv._near_future_active_colors(board))
        fire = np.array([result.values[k].score for k in iv.NEAR_FUTURE_K_LEVELS])
    return np.r_[shape, fire]


def arrival_features(sides: Array, firing: NDArray, source_side: int = 0) -> Array:
    """絶対量は側交換のみ、差分だけ符号反転する。側番号0=1P、1=2P。"""
    own, opp = np.asarray(sides)[[source_side, 1 - source_side]]
    triggers = np.asarray(firing)[[source_side, 1 - source_side]]
    return np.r_[own, opp, own - opp, triggers]


def sent_features(sent: Array, missing: NDArray, source_side: int = 0) -> Array:
    """送り量を有界化する。差分列のみ符号付き、欠測はNaNのまま維持。"""
    own, opp = np.asarray(sent)[[source_side, 1 - source_side]]
    net = own - opp
    more = [float(net > 0), float(net < 0)] if np.isfinite(net) else [np.nan, np.nan]
    return np.r_[own / (1 + own), opp / (1 + opp), net / (1 + abs(net)),
                 more, np.asarray(missing)[[source_side, 1 - source_side]].astype(float)]


def score_features(before: Array, after: Array, elapsed_sec: float,
                   source_side: int = 0, from_first_move: bool = False) -> Array:
    """OCR差分を既存マージンレートで換算する。小数送り量はT11に合わせる。"""
    validate_elapsed(elapsed_sec)
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.shape != (2,) or after.shape != (2,) or source_side not in (0, 1):
        raise ValueError("得点は1P/2Pの2要素、source_sideは0または1が必要")
    missing = ~np.isfinite(before) | ~np.isfinite(after) | (before < 0) | (after < 0)
    rate = compute_effective_rate(elapsed_sec, from_first_move=from_first_move)
    sent = np.where(missing, np.nan, np.maximum(after - before, 0) / rate)
    return sent_features(sent, missing, source_side)
