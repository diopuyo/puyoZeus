"""collect_boards_lean.py のユニットテスト。

テスト方針:
- 動画認識・重い collect 実行は一切しない。
- 合成 Board / 合成スナップショット挿入で内部ロジックのみ検証する。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    Board,
)
from src.board_state_machine import BoardState


# ============================
# テスト用ヘルパ
# ============================

def _make_board(color: int = COLOR_RED, row_count: int = 2) -> Board:
    """下段 row_count 行を color で埋めた合成盤面を返す。"""
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for col in range(BOARD_COLS):
        for row_offset in range(row_count):
            g[BOARD_ROWS - 1 - row_offset][col] = color
    return Board.from_list(g)


def _make_empty_board() -> Board:
    """全セルが EMPTY (0) の盤面を返す。"""
    return Board.from_list([[0] * BOARD_COLS for _ in range(BOARD_ROWS)])


def _import_lean():
    """collect_boards_lean モジュールをインポートして返す。"""
    import scripts.collect_boards_lean as mod
    return mod


# ============================
# _LeanNpzAccumulator のテスト
# ============================

class TestLeanNpzAccumulator:
    """_LeanNpzAccumulator の append / save / won 付与 を検証する。"""

    def test_empty_save_creates_npz(self, tmp_path: Path) -> None:
        """空バッファを保存しても npz が生成されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        out = tmp_path / "empty.npz"
        acc.save(out)
        assert out.exists()
        data = np.load(str(out), allow_pickle=True)
        assert "grids" in data
        assert "won" in data

    def test_append_single_shape(self, tmp_path: Path) -> None:
        """1 件 append → npz grids が (1, 13, 6) int8 になること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v29", "1P", 12.5, 0, 100)
        out = tmp_path / "single.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["grids"].shape == (1, BOARD_ROWS, BOARD_COLS)
        assert data["grids"].dtype == np.int8

    def test_npz_keys_present(self, tmp_path: Path) -> None:
        """npz に必須キーが全て存在すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "keys.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        # collect_indicators_v2 --board-npz との互換キー
        for key in ("grids", "video_id", "side", "t_sec", "game_idx", "frame_idx"):
            assert key in data, f"キー '{key}' が npz に存在しない"
        # lean 追加キー
        assert "won" in data

    def test_won_initial_nan(self, tmp_path: Path) -> None:
        """append 直後は won=NaN で仮置きされること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "nan.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert math.isnan(float(data["won"][0]))

    def test_assign_won_1p_wins(self, tmp_path: Path) -> None:
        """1P の最終 score が大きい場合、1P 盤面は won=1.0 になること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 10)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 1.0, 0, 10)
        # 1P が score 5000 > 2P score 3000 → 1P 勝ち
        game_final = {0: {"1P": 5000, "2P": 3000}}
        acc.assign_won_labels(game_final)
        out = tmp_path / "p1wins.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        sides = data["side"].tolist()
        wons = data["won"].tolist()
        p1_won = wons[sides.index("1P")]
        p2_won = wons[sides.index("2P")]
        assert float(p1_won) == 1.0
        assert float(p2_won) == 0.0

    def test_assign_won_2p_wins(self, tmp_path: Path) -> None:
        """2P の最終 score が大きい場合、1P 盤面は won=0.0 になること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 2.0, 0, 20)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 2.0, 0, 20)
        game_final = {0: {"1P": 2000, "2P": 8000}}
        acc.assign_won_labels(game_final)
        out = tmp_path / "p2wins.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        sides = data["side"].tolist()
        wons = data["won"].tolist()
        p1_won = wons[sides.index("1P")]
        assert float(p1_won) == 0.0

    def test_assign_won_equal_score_is_nan(self) -> None:
        """両者 score が同一の場合は won=NaN のまま (判定不能)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        game_final = {0: {"1P": 4000, "2P": 4000}}
        acc.assign_won_labels(game_final)
        assert math.isnan(acc.wons[0])

    def test_assign_won_multiple_games(self) -> None:
        """複数 game_idx に対して正しく won が付与されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # game 0: 1P 勝ち
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 1.0, 0, 1)
        # game 1: 2P 勝ち
        acc.append(_make_board(COLOR_GREEN)._grid, "v29", "1P", 60.0, 1, 600)
        acc.append(_make_board(COLOR_YELLOW)._grid, "v29", "2P", 60.0, 1, 600)
        game_final = {
            0: {"1P": 5000, "2P": 3000},
            1: {"1P": 1000, "2P": 9000},
        }
        acc.assign_won_labels(game_final)
        # game 0 の 1P snapshot
        assert float(acc.wons[0]) == 1.0
        assert float(acc.wons[1]) == 0.0
        # game 1 の 1P snapshot
        assert float(acc.wons[2]) == 0.0
        assert float(acc.wons[3]) == 1.0

    def test_grid_copy_independence(self) -> None:
        """append 後に元 grid を書き換えても蓄積値が変わらないこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v29", "1P", 1.0, 0, 1)
        board._grid[12, 0] = COLOR_BLUE
        # acc の中身は append 時点のコピー → COLOR_RED のまま
        assert int(acc.grids[0][12, 0]) == COLOR_RED


# ============================
# _SideState / ゲーム境界のテスト
# ============================

class TestSideState:
    """_SideState と _update_game_boundary のロジックを検証する。"""

    def test_game_idx_increments_on_score_reset(self) -> None:
        """score が SCORE_RESET_THRESHOLD 以上減少したら game_idx が増えること。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_game_boundary(state, 5000)
        mod._update_game_boundary(state, 100)  # 4900 減少 → 境界
        assert state.game_idx == 1

    def test_game_idx_no_increment_on_small_decrease(self) -> None:
        """score 減少が SCORE_RESET_THRESHOLD 未満なら game_idx は変わらないこと。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_game_boundary(state, 5000)
        mod._update_game_boundary(state, 4600)  # 400 減少 < 500
        assert state.game_idx == 0

    def test_final_scores_tracked(self) -> None:
        """各 game_idx の最終 score が正しく記録されること。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_game_boundary(state, 1000)
        mod._update_game_boundary(state, 3000)
        mod._update_game_boundary(state, 0)    # score リセット → game 1 へ
        mod._update_game_boundary(state, 2000)
        # game 0 の最終 score は 3000 (リセット直前)
        # game 1 の最終 score は 2000
        assert state.final_scores[0] == 0      # update は score=0 を記録してからゲーム境界
        assert state.final_scores.get(1) == 2000


# ============================
# _should_emit のテスト
# ============================

class TestShouldEmit:
    """_should_emit の間引きと全消し除外を検証する。"""

    def test_stable_nonempty_emits(self) -> None:
        """STABLE かつ盤面非空なら emit される。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        assert mod._should_emit(state, board, BoardState.STABLE)

    def test_non_stable_not_emitted(self) -> None:
        """STABLE 以外の状態は emit されない。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        for s in (BoardState.TSUMO_FALL, BoardState.CHAIN, BoardState.MENU):
            assert not mod._should_emit(state, board, s)

    def test_empty_board_not_emitted(self) -> None:
        """全消し直後の空盤面は emit されない。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_empty_board()
        assert not mod._should_emit(state, board, BoardState.STABLE)

    def test_same_grid_deduplicated(self) -> None:
        """直前と同一盤面は間引かれること。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        # 1 回目: emit される
        assert mod._should_emit(state, board, BoardState.STABLE)
        # last_emitted_grid を設定して 2 回目をシミュレート
        state.last_emitted_grid = board._grid.tobytes()
        # 2 回目: 同一 grid なので emit されない
        assert not mod._should_emit(state, board, BoardState.STABLE)

    def test_different_grid_after_dedup_emits(self) -> None:
        """間引き後に異なる盤面が来たら再び emit される。"""
        mod = _import_lean()
        state = mod._SideState()
        board_r = _make_board(COLOR_RED)
        state.last_emitted_grid = board_r._grid.tobytes()
        board_b = _make_board(COLOR_BLUE)
        assert mod._should_emit(state, board_b, BoardState.STABLE)

    def test_none_board_not_emitted(self) -> None:
        """board=None は emit されない。"""
        mod = _import_lean()
        state = mod._SideState()
        assert not mod._should_emit(state, None, BoardState.STABLE)  # type: ignore


# ============================
# _merge_final_scores のテスト
# ============================

class TestMergeFinalScores:
    """_merge_final_scores の統合ロジックを検証する。"""

    def test_merge_both_sides(self) -> None:
        """1P/2P の final_scores が同じ game_idx で正しく統合されること。"""
        mod = _import_lean()
        state_p1 = mod._SideState()
        state_p2 = mod._SideState()
        state_p1.final_scores = {0: 5000, 1: 3000}
        state_p2.final_scores = {0: 4000, 1: 7000}
        merged = mod._merge_final_scores(state_p1, state_p2)
        assert merged[0]["1P"] == 5000
        assert merged[0]["2P"] == 4000
        assert merged[1]["1P"] == 3000
        assert merged[1]["2P"] == 7000

    def test_merge_missing_side_is_none(self) -> None:
        """一方の side にしか game_idx がない場合、他方は None になること。"""
        mod = _import_lean()
        state_p1 = mod._SideState()
        state_p2 = mod._SideState()
        state_p1.final_scores = {0: 5000}
        state_p2.final_scores = {}  # 2P は game 0 のデータなし
        merged = mod._merge_final_scores(state_p1, state_p2)
        assert merged[0]["1P"] == 5000
        assert merged[0]["2P"] is None


# ============================
# build_board_pairs との整合性テスト
# ============================

class TestNpzCompatibility:
    """lean npz が build_board_pairs の load_npz_dir で読み込めることを確認する。"""

    def test_lean_npz_loadable_by_build_pairs(self, tmp_path: Path) -> None:
        """lean 出力 npz が build_board_pairs.load_npz_dir で読み込めること。"""
        mod = _import_lean()
        import scripts.build_board_pairs as pairs_mod

        # lean npz を手動で生成
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 10.0, 0, 100)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 10.0, 0, 100)
        game_final = {0: {"1P": 5000, "2P": 3000}}
        acc.assign_won_labels(game_final)

        npz_dir = tmp_path / "npz"
        npz_dir.mkdir()
        acc.save(npz_dir / "v29.npz")

        # build_board_pairs が読み込めること (例外なし)
        df, grids = pairs_mod.load_npz_dir(npz_dir)
        assert len(df) == 2
        assert grids.shape == (2, BOARD_ROWS, BOARD_COLS)
        # 必須カラムの確認
        for col in ("video_id", "side", "t_sec", "game_idx", "frame_idx"):
            assert col in df.columns

    def test_lean_npz_grid_shape(self, tmp_path: Path) -> None:
        """lean npz の grids が (N, 13, 6) int8 であること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        for i in range(5):
            acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", float(i), 0, i * 10)
        out = tmp_path / "shape_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["grids"].shape == (5, BOARD_ROWS, BOARD_COLS)
        assert data["grids"].dtype == np.int8

    def test_won_array_float32(self, tmp_path: Path) -> None:
        """won 配列が float32 であること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        acc.assign_won_labels({0: {"1P": 5000, "2P": 3000}})
        out = tmp_path / "dtype.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["won"].dtype == np.float32
