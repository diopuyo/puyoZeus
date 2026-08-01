"""Step6 (2026-08-01): scripts/augment_exchange_labels_with_sim.py の回帰テスト。

軽量な手作り NpzRecord のみを使用し、実動画・実npzの重い処理は行わない
(tests/test_exchange_effectiveness_step2.py と同じ方針)。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.board import Board
from scripts.label_exchange_outcome import NpzRecord
from scripts.measure_exchange_dynamics import OppCoverageStatus
from scripts.augment_exchange_labels_with_sim import (
    _VideoCache,
    _compute_sim_columns_for_row,
    _opp_side,
    _video_id_to_npz_stem,
)


# ============================
# 小道具: 合成 NpzRecord
# ============================

def _empty_grid() -> np.ndarray:
    """13x6 の空盤面グリッド。"""
    return np.zeros((13, 6), dtype=np.int8)


def _make_records(
    fire_scores: list[int],
    fire_t: list[float],
    opp_scores: list[int],
    opp_t: list[float],
    game_idx_fire: "list[int] | None" = None,
    game_idx_opp: "list[int] | None" = None,
) -> list[NpzRecord]:
    """1P=fire_side, 2P=opp_side の合成 NpzRecord ペアを作る (全フレーム空盤面)。"""
    n_fire = len(fire_scores)
    n_opp = len(opp_scores)
    gi_fire = game_idx_fire if game_idx_fire is not None else [0] * n_fire
    gi_opp = game_idx_opp if game_idx_opp is not None else [0] * n_opp
    grids_fire = np.stack([_empty_grid() for _ in range(n_fire)])
    grids_opp = np.stack([_empty_grid() for _ in range(n_opp)])
    rec_fire = NpzRecord(
        video_id="video_test1", side="1P",
        t_sec=np.array(fire_t, dtype=np.float32),
        game_idx=np.array(gi_fire, dtype=np.int32),
        grids=grids_fire,
        won=np.zeros(n_fire, dtype=np.float32),
        score=np.array(fire_scores, dtype=np.int32),
    )
    rec_opp = NpzRecord(
        video_id="video_test1", side="2P",
        t_sec=np.array(opp_t, dtype=np.float32),
        game_idx=np.array(gi_opp, dtype=np.int32),
        grids=grids_opp,
        won=np.zeros(n_opp, dtype=np.float32),
        score=np.array(opp_scores, dtype=np.int32),
    )
    return [rec_fire, rec_opp]


def _make_row(t_sec: float, game_idx: int, fire_side: str, approx_fire_chains: float) -> "pd.Series":
    return pd.Series({
        "t_sec": t_sec, "game_idx": game_idx,
        "fire_side": fire_side, "approx_fire_chains": approx_fire_chains,
    })


# ============================
# _video_id_to_npz_stem / _opp_side
# ============================

def test_video_id_to_npz_stem_strips_prefix() -> None:
    """CSV の video_id ("video_c10") から npz ファイル名 stem ("c10") へ変換する。"""
    assert _video_id_to_npz_stem("video_c10") == "c10"


def test_video_id_to_npz_stem_passthrough_without_prefix() -> None:
    """既に接頭辞が無い場合はそのまま返す (想定外形式への安全策)。"""
    assert _video_id_to_npz_stem("c10") == "c10"


def test_opp_side_mapping() -> None:
    assert _opp_side("1P") == "2P"
    assert _opp_side("2P") == "1P"


# ============================
# _VideoCache: 遅延構築・キャッシュ
# ============================

def test_video_cache_boards_for_side_lazy_and_cached() -> None:
    """boards_for_side は初回だけ構築し、以降は同じオブジェクトを返す(再構築しない)。"""
    records = _make_records(
        fire_scores=[0, 700], fire_t=[0.0, 1.0],
        opp_scores=[0, 0], opp_t=[0.0, 1.0],
    )
    cache = _VideoCache(records)
    boards_1 = cache.boards_for_side("1P")
    boards_2 = cache.boards_for_side("1P")
    assert boards_1 is boards_2  # 同一オブジェクト (再構築していない証拠)
    assert len(boards_1) == 2


def test_video_cache_boards_for_side_missing_side_returns_none() -> None:
    """存在しない side を渡すと None を返す。"""
    records = _make_records(
        fire_scores=[0, 700], fire_t=[0.0, 1.0],
        opp_scores=[0, 0], opp_t=[0.0, 1.0],
    )
    cache = _VideoCache(records)
    assert cache.boards_for_side("3P") is None


# ============================
# _compute_sim_columns_for_row
# ============================

def test_opp_chaining_scenario_yields_zero_expected_counter() -> None:
    """相手側スコアが大きなギャップを挟んで跳ねる (OPP_CHAINING 相当) 場合、

    sim_expected_counter_ojama は 0 に近くなる
    (measure_exchange_dynamics._classify_opp_coverage の判定を流用)。

    ⚠️ _restrict_to_time_window は攻撃側自身のこのゲームの実時刻範囲
    (fire_t の min〜max) で相手フレームを絞り込むため、相手の「連鎖中ギャップ」
    (t=0〜20) をこの範囲内に収める必要がある (fire_t に t=20.0 のダミー
    フレームを含めて own_game_end_t=20.0 まで広げている)。
    """
    records = _make_records(
        fire_scores=[0, 700, 700], fire_t=[0.0, 5.0, 20.0],
        opp_scores=[0, 5000], opp_t=[0.0, 20.0],  # 20秒ギャップ+跳ね=OPP_CHAINING
    )
    cache = _VideoCache(records)
    row = _make_row(t_sec=5.0, game_idx=0, fire_side="1P", approx_fire_chains=1.0)
    k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, mode="precise")
    assert not math.isnan(k_hands)
    assert exp_counter == pytest.approx(0.0)
    # attacker_ojama_sent = 700 // 70 = 10、期待反撃0なのでnet=10 -> ojama_damageは正の値
    assert 0.0 <= damage <= 1.0


def test_delta_score_unavailable_returns_all_nan() -> None:
    """発火フレームが先頭 (直前有効スコア無し) の場合は3値ともNaNを返す。"""
    records = _make_records(
        fire_scores=[700], fire_t=[0.0],  # 先頭フレームしか無い = 直前スコア無し
        opp_scores=[0], opp_t=[0.0],
    )
    cache = _VideoCache(records)
    row = _make_row(t_sec=0.0, game_idx=0, fire_side="1P", approx_fire_chains=1.0)
    k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, mode="precise")
    assert math.isnan(k_hands)
    assert math.isnan(exp_counter)
    assert math.isnan(damage)


def test_unknown_game_idx_returns_all_nan() -> None:
    """CSV行の game_idx が npz 側に存在しない場合は3値ともNaNを返す

    (壊れたデータ・突き合わせ失敗を 0 に丸めない)。
    """
    records = _make_records(
        fire_scores=[0, 700], fire_t=[0.0, 1.0],
        opp_scores=[0, 0], opp_t=[0.0, 1.0],
    )
    cache = _VideoCache(records)
    row = _make_row(t_sec=1.0, game_idx=99, fire_side="1P", approx_fire_chains=1.0)
    k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, mode="precise")
    assert math.isnan(k_hands)
    assert math.isnan(exp_counter)
    assert math.isnan(damage)


def test_observed_scenario_computes_finite_damage() -> None:
    """相手フレームが密 (ギャップ<=2秒、OBSERVED相当) なら通常通り

    expected_fire_power 経由の期待反撃量が計算され、有限値を返す。
    """
    records = _make_records(
        fire_scores=[0, 700], fire_t=[0.0, 5.0],
        # 5.0 の前後に密な観測点を置く (idx_before=4.5, idx_after=5.5、
        # ギャップ1.0秒<=2.0秒 => OBSERVED)。
        opp_scores=[0, 0, 0], opp_t=[0.0, 4.5, 5.5],
    )
    cache = _VideoCache(records)
    row = _make_row(t_sec=5.0, game_idx=0, fire_side="1P", approx_fire_chains=3.0)
    k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, mode="precise")
    assert not math.isnan(k_hands)
    assert not math.isnan(exp_counter)
    assert not math.isnan(damage)
    assert 0.0 <= damage <= 1.0
