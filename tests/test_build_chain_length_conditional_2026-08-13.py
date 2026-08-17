"""scripts/_build_chain_length_conditional_2026-08-13.py の単体テスト。

#3 修正 (--counter-remaining-time) が使う E[最終連鎖数|観測N到達] テーブル
生成スクリプトの正当性を検証する: (a) ヒストグラム→条件付き期待値の計算式
(b) npz からの発火前盤面の再構成 (誤検出/重複タグの回避を含む)。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_RED


@pytest.fixture(scope="module")
def mod():
    """ハイフン入りファイル名のためモジュールとして直接ロードする
    (tests/test_viz_flag_defaults_match_library.py と同じ方式)。"""
    path = Path(__file__).resolve().parent.parent / "scripts" / (
        "_build_chain_length_conditional_2026-08-13.py"
    )
    spec = importlib.util.spec_from_file_location("_chain_len_cond_for_test", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["_chain_len_cond_for_test"] = m
    spec.loader.exec_module(m)
    return m


# ============================
# build_conditional_table (純粋関数)
# ============================


def test_build_conditional_table_matches_hand_calculation(mod) -> None:
    """K=1が2件、K=2が1件、K=3が1件のとき、E[final|1]=(1*2+2*1+3*1)/4=1.75、
    E[final|2]=(2*1+3*1)/2=2.5、E[final|3]=3.0/1=3.0。"""
    table = mod.build_conditional_table([1, 1, 2, 3])
    result = table["expected_final_given_reached_n"]
    assert result["1"] == pytest.approx(1.75)
    assert result["2"] == pytest.approx(2.5)
    assert result["3"] == pytest.approx(3.0)
    assert table["total_events"] == 4
    assert table["max_chain_count_observed"] == 3
    assert table["histogram"] == {"1": 2, "2": 1, "3": 1}


def test_build_conditional_table_empty_input(mod) -> None:
    table = mod.build_conditional_table([])
    assert table["total_events"] == 0
    assert table["max_chain_count_observed"] == 0
    assert table["expected_final_given_reached_n"] == {}


def test_build_conditional_table_expected_is_monotonic_non_decreasing(mod) -> None:
    """N が大きいほど E[final|N] も大きい (小さい連鎖が足切りされていくため)。"""
    table = mod.build_conditional_table([1, 2, 2, 3, 4, 5, 5, 5])
    values = table["expected_final_given_reached_n"]
    keys_sorted = sorted(int(k) for k in values)
    seq = [values[str(k)] for k in keys_sorted]
    assert seq == sorted(seq)


# ============================
# _find_before_board_index
# ============================


def test_find_before_board_index_finds_nearest_drop(mod) -> None:
    # nz列: idx0=56, idx1=45(56->45で11減少=erasure), idx2=47, idx3=49
    nz_counts = [56, 45, 47, 49]
    idxs = [0, 1, 2, 3]
    # pos=3 (タグ行) から遡ると j=1 で 56-45=11>=4 の減少が見つかる -> idxs[0]
    assert mod._find_before_board_index(nz_counts, idxs, pos=3) == 0


def test_find_before_board_index_returns_none_when_no_drop(mod) -> None:
    nz_counts = [10, 12, 14, 16]  # 単調増加 (減少なし)
    idxs = [0, 1, 2, 3]
    assert mod._find_before_board_index(nz_counts, idxs, pos=3) is None


def test_find_before_board_index_ignores_small_decrease_below_threshold(mod) -> None:
    # ERASURE_MIN_DROP=4 未満の減少 (通常運転のブレ) は無視する
    nz_counts = [10, 8, 8, 8]  # 10->8 は2減少のみ (閾値未満)
    idxs = [0, 1, 2, 3]
    assert mod._find_before_board_index(nz_counts, idxs, pos=3) is None


# ============================
# _chain_counts_in_file (npz からの再構成、end-to-end)
# ============================


def _make_before_chain_grid() -> np.ndarray:
    """4個横並びの赤 (1連鎖で消える最小盤面) を最下段に置いた発火前盤面。"""
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    grid[BOARD_ROWS - 1, 0:4] = COLOR_RED
    return grid


def test_chain_counts_in_file_reconstructs_single_chain(mod, tmp_path: Path) -> None:
    """タグ行の直前行が既に発火後 (実データで判明した off-by-one) の
    ケースを模擬し、正しく1つ前まで遡って before_board を特定できること。
    """
    before = _make_before_chain_grid()
    after = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)  # 4個消えて空

    # row0=発火前(nz=4) / row1=発火後(nz=0、まだタグなし)
    # / row2=次のツモ設置後(nz=2、ここにタグが付く=実データの off-by-one を再現)
    grids = np.stack([before, after, after.copy(), before.copy() * 0])
    grids[2, BOARD_ROWS - 1, 4] = COLOR_RED
    grids[2, BOARD_ROWS - 1, 5] = COLOR_RED  # nz=2 (次のツモ)

    npz_path = tmp_path / "fake.npz"
    np.savez(
        npz_path,
        grids=grids[:3],
        side=np.array(["1P", "1P", "1P"]),
        game_idx=np.array([0, 0, 0], dtype=np.int32),
        t_sec=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        chain_trigger_sec=np.array([np.nan, np.nan, 0.15], dtype=np.float32),
        chain_mechanism=np.array(["", "", "baseline"]),
    )
    counts = mod._chain_counts_in_file(npz_path)
    assert counts == [1]


def test_chain_counts_in_file_deduplicates_repeated_trigger_sec(mod, tmp_path: Path) -> None:
    """同一 trigger_sec が連続する行 (hold窓での重複タグ、実データで確認済み)
    は1回だけ数える。"""
    before = _make_before_chain_grid()
    after = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    after_plus_one = after.copy()
    after_plus_one[BOARD_ROWS - 1, 4] = COLOR_RED

    grids = np.stack([before, after, after_plus_one])
    npz_path = tmp_path / "fake_dup.npz"
    np.savez(
        npz_path,
        grids=grids,
        side=np.array(["1P", "1P", "1P"]),
        game_idx=np.array([0, 0, 0], dtype=np.int32),
        t_sec=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        # row1/row2 が同一 trigger_sec (重複タグ) を持つ
        chain_trigger_sec=np.array([np.nan, 0.05, 0.05], dtype=np.float32),
        chain_mechanism=np.array(["", "baseline", "baseline"]),
    )
    counts = mod._chain_counts_in_file(npz_path)
    assert counts == [1]  # 2回でなく1回


def test_chain_counts_in_file_skips_zero_chain_noise(mod, tmp_path: Path) -> None:
    """発火前盤面を simulate しても連鎖が起きない (誤検出) 場合はスキップする。"""
    no_chain_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    no_chain_board[BOARD_ROWS - 1, 0] = COLOR_RED  # 1個だけ、消えない
    after = no_chain_board.copy()
    after[BOARD_ROWS - 1, 1] = COLOR_RED  # 1個増えただけ (nz増加、減少なし)

    grids = np.stack([no_chain_board, after])
    npz_path = tmp_path / "fake_noise.npz"
    np.savez(
        npz_path,
        grids=grids,
        side=np.array(["1P", "1P"]),
        game_idx=np.array([0, 0], dtype=np.int32),
        t_sec=np.array([0.0, 0.1], dtype=np.float32),
        chain_trigger_sec=np.array([np.nan, 0.05], dtype=np.float32),
        chain_mechanism=np.array(["", "baseline"]),
    )
    counts = mod._chain_counts_in_file(npz_path)
    assert counts == []


def test_chain_counts_in_file_missing_columns_returns_empty(mod, tmp_path: Path) -> None:
    """chain_trigger_sec/chain_mechanism 列が無い npz (旧収集) は空リスト。"""
    grids = np.zeros((1, BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    npz_path = tmp_path / "no_chain_cols.npz"
    np.savez(
        npz_path, grids=grids,
        side=np.array(["1P"]), game_idx=np.array([0], dtype=np.int32),
        t_sec=np.array([0.0], dtype=np.float32),
    )
    assert mod._chain_counts_in_file(npz_path) == []
