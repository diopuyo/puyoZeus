"""src/chain_bitboard.py の正当性回帰テスト。

既存 ChainSimulator (BFS方式、src/chain.py) との完全一致を保証する。
速度の議論より前に必ずこのテストが全パスすること (コーディネータ方針)。

`src/chain.py` の ChainSimulator は一切変更しないため、本テストは
既存テストスイートに影響を与えない (新規追加のみ)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import (
    Board,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_OJAMA,
)
from src.chain import ChainSimulator
from src.chain_bitboard import (
    batch_adjacency_tiebreak,
    board_to_planes,
    planes_to_board,
    simulate_single,
)

BOARDS_DIR = Path("data/indicators_v2/boards")


def _sim() -> ChainSimulator:
    return ChainSimulator()


def _assert_matches(board: Board) -> None:
    """既存 ChainSimulator と chain_bitboard の結果が完全一致することを確認する。"""
    sim = _sim()
    expected = sim.simulate(board)
    got = simulate_single(board)

    assert expected.chain_count == got.chain_count, (
        f"chain_count 不一致: expected={expected.chain_count} got={got.chain_count}"
    )
    assert expected.total_erased == got.total_erased, (
        f"total_erased 不一致: expected={expected.total_erased} got={got.total_erased}"
    )
    assert expected.total_ojama == got.total_ojama, (
        f"total_ojama 不一致: expected={expected.total_ojama} got={got.total_ojama}"
    )
    got_final_board = planes_to_board(got.final_planes)
    assert np.array_equal(expected.final_board._grid, got_final_board._grid), (
        "final_board 不一致 (グリッド差分あり)"
    )


# ============================
# round-trip 変換
# ============================


def test_board_to_planes_roundtrip_empty() -> None:
    board = Board()
    planes = board_to_planes(board)
    back = planes_to_board(planes)
    assert np.array_equal(board._grid, back._grid)


def test_board_to_planes_roundtrip_random_sample() -> None:
    """v29.npz からサンプルした盤面で round-trip 変換が完全に元に戻ることを確認する。"""
    npz_path = BOARDS_DIR / "v29.npz"
    if not npz_path.exists():
        pytest.skip(f"{npz_path} が存在しない (npz キャッシュ未生成)")
    data = np.load(str(npz_path), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(0)
    n = min(20, len(grids))
    idxs = rng.choice(len(grids), size=n, replace=False)
    for i in idxs:
        board = Board.from_list(grids[i].tolist())
        planes = board_to_planes(board)
        back = planes_to_board(planes)
        assert np.array_equal(board._grid, back._grid), f"round-trip失敗 idx={i}"


# ============================
# 手作り境界ケース
# ============================


def test_empty_board_no_chain() -> None:
    _assert_matches(Board())


def test_vertical_four_including_hidden_row() -> None:
    """row0(隠し段)を含む縦4連結が正しく消えることを確認する (ama get_mask_12制限を採用しない根拠)。"""
    board = Board()
    for row in range(4):
        board.set(row, 0, COLOR_RED)
    _assert_matches(board)


def test_ojama_adjacent_clear() -> None:
    """4連結消去に隣接するお邪魔も同時消去されることを確認する。"""
    board = Board()
    board.set(9, 1, COLOR_BLUE)
    board.set(9, 2, COLOR_BLUE)
    board.set(9, 3, COLOR_BLUE)
    board.set(8, 2, COLOR_BLUE)
    board.set(10, 1, COLOR_OJAMA)
    board.set(10, 2, COLOR_OJAMA)
    board.set(10, 3, COLOR_OJAMA)
    board.set(7, 2, COLOR_OJAMA)
    _assert_matches(board)


def test_multi_step_chain_two_steps() -> None:
    """重力による2連鎖 (1段目消去→落下→2段目消去) が正しく判定されることを確認する。"""
    board = Board()
    board.set(8, 0, COLOR_BLUE)
    for row in range(9, 13):
        board.set(row, 0, COLOR_RED)
    for row in range(10, 13):
        board.set(row, 1, COLOR_BLUE)
    result = ChainSimulator().simulate(board)
    assert result.chain_count == 2, "テスト構築ミス: 2連鎖になっていない"
    _assert_matches(board)


def test_checkerboard_no_ignition() -> None:
    """市松模様 (どの色も4連結にならない) は連鎖0を保つことを確認する。"""
    grid = [
        [COLOR_RED if (r + c) % 2 == 0 else COLOR_BLUE for c in range(6)]
        for r in range(13)
    ]
    board = Board.from_list(grid)
    _assert_matches(board)


def test_full_board_l_shape_tetromino() -> None:
    """L字4連結 (3連結の隣に垂直方向の隅) が正しく消去されることを確認する。

    ama の m2/m3 定式が直線・L字いずれの形状でも正しく機能するかの
    境界ケース (フェーズ0裏取りで手計算検証済み)。
    """
    board = Board()
    board.set(12, 0, COLOR_GREEN)
    board.set(12, 1, COLOR_GREEN)
    board.set(12, 2, COLOR_GREEN)
    board.set(11, 2, COLOR_GREEN)
    _assert_matches(board)


# ============================
# 実データ regression (複数動画・複数盤面)
# ============================


@pytest.mark.parametrize("video_id", ["v29", "v30", "v31", "v32", "v33"])
def test_matches_chain_simulator_on_real_boards(video_id: str) -> None:
    """実際の STABLE 盤面サンプルで既存 ChainSimulator と完全一致することを確認する。"""
    npz_path = BOARDS_DIR / f"{video_id}.npz"
    if not npz_path.exists():
        pytest.skip(f"{npz_path} が存在しない (npz キャッシュ未生成)")
    data = np.load(str(npz_path), allow_pickle=True)
    grids = data["grids"]
    rng = np.random.default_rng(42)
    n = min(30, len(grids))
    if n == 0:
        pytest.skip("盤面サンプルが空")
    idxs = rng.choice(len(grids), size=n, replace=False)
    for i in idxs:
        board = Board.from_list(grids[i].tolist())
        if board.is_dead():
            continue
        _assert_matches(board)


# ============================
# batch_adjacency_tiebreak (near_future_fire_power 同点タイブレーク用, 2026-08-09)
# ============================


def test_batch_adjacency_tiebreak_empty_boards_list() -> None:
    """空リストを渡すと長さ0の配列を返すこと (境界ケース)。"""
    pair_proxy, triple_proxy = batch_adjacency_tiebreak([])
    assert pair_proxy.shape == (0,)
    assert triple_proxy.shape == (0,)


def test_batch_adjacency_tiebreak_empty_board_is_zero() -> None:
    """空盤面は隣接ペア・近傍2方向以上のセルともに0であること。"""
    pair_proxy, triple_proxy = batch_adjacency_tiebreak([Board()])
    assert pair_proxy[0] == 0
    assert triple_proxy[0] == 0


def test_batch_adjacency_tiebreak_isolated_cells_are_zero() -> None:
    """互いに隣接しない同色セルのみの盤面は pair_proxy が0であること。"""
    board = Board()
    board.set(12, 0, COLOR_RED)
    board.set(12, 2, COLOR_RED)
    board.set(12, 4, COLOR_RED)
    pair_proxy, triple_proxy = batch_adjacency_tiebreak([board])
    assert pair_proxy[0] == 0
    assert triple_proxy[0] == 0


def test_batch_adjacency_tiebreak_l_shape_has_pair_and_triple() -> None:
    """L字3連結 (あと1個で消える形) は pair_proxy>0・triple_proxy>0 になること。"""
    board = Board()
    board.set(12, 0, COLOR_GREEN)
    board.set(12, 1, COLOR_GREEN)
    board.set(11, 0, COLOR_GREEN)
    pair_proxy, triple_proxy = batch_adjacency_tiebreak([board])
    assert pair_proxy[0] == 2  # (12,0)-(12,1) の横エッジ + (12,0)-(11,0) の縦エッジ
    assert triple_proxy[0] == 1  # (12,0) が2方向に同色近傍を持つ


def test_batch_adjacency_tiebreak_batch_matches_single_calls() -> None:
    """複数盤面をバッチで渡した結果が、1件ずつ渡した結果の連結と一致すること。"""
    isolated = Board()
    isolated.set(12, 0, COLOR_RED)

    l_shape = Board()
    l_shape.set(12, 0, COLOR_GREEN)
    l_shape.set(12, 1, COLOR_GREEN)
    l_shape.set(11, 0, COLOR_GREEN)

    batch_pair, batch_triple = batch_adjacency_tiebreak([isolated, l_shape])
    single_pair_0, single_triple_0 = batch_adjacency_tiebreak([isolated])
    single_pair_1, single_triple_1 = batch_adjacency_tiebreak([l_shape])

    assert batch_pair[0] == single_pair_0[0]
    assert batch_triple[0] == single_triple_0[0]
    assert batch_pair[1] == single_pair_1[0]
    assert batch_triple[1] == single_triple_1[0]


def test_batch_adjacency_tiebreak_does_not_mutate_boards() -> None:
    """stateless 原則: 呼出前後で盤面が変化しないこと。"""
    board = Board()
    board.set(12, 0, COLOR_BLUE)
    board.set(12, 1, COLOR_BLUE)
    before = board.copy()
    batch_adjacency_tiebreak([board])
    assert np.array_equal(board._grid, before._grid)
