"""collect_boards_lean.py のユニットテスト。

テスト方針:
- 動画認識・重い collect 実行は一切しない。
- 合成 Board / 合成スナップショット挿入で内部ロジックのみ検証する。
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_YELLOW,
    Board,
)
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline


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
        # collect_indicators_v2 --board-npz との互換キー (後方互換: 既存キーは維持)
        for key in ("grids", "video_id", "side", "t_sec", "game_idx", "frame_idx"):
            assert key in data, f"キー '{key}' が npz に存在しない"
        # lean 追加キー
        assert "won" in data
        # 修正1追加キー: score (オフライン再ラベル付け用)
        assert "score" in data

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
        """各 game_idx の最終 score が正しく記録されること。

        修正後: リセット発生時に旧ゲームの最終 score = リセット直前の高値(3000)を記録する。
        旧実装はリセット後低値(0)を記録するバグがあり、勝者判定が全て None になっていた。
        """
        mod = _import_lean()
        state = mod._SideState()
        mod._update_game_boundary(state, 1000)
        mod._update_game_boundary(state, 3000)
        mod._update_game_boundary(state, 0)    # score リセット → game 1 へ
        mod._update_game_boundary(state, 2000)
        # 修正後: game 0 の最終 score = リセット直前の高値 3000
        assert state.final_scores[0] == 3000
        # game 1 の最終 score は 2000
        assert state.final_scores.get(1) == 2000

    def test_pre_reset_high_value_captured(self) -> None:
        """リセット検知時に旧ゲームの最終スコアがリセット直前の高値で記録されること。

        バグ修正の核心: 旧実装は final_scores[game_idx] = score (≈0) を書いてから
        game_idx を進めていた。修正後は prev_score (高値) を記録してから進める。
        """
        mod = _import_lean()
        state = mod._SideState()
        # ゲーム0 のスコア上昇
        for s in [100, 500, 1200, 4800, 7500]:
            mod._update_game_boundary(state, s)
        # リセット: 7500 → 50 (差 7450 >= 500 threshold)
        mod._update_game_boundary(state, 50)
        # ゲーム0 の最終スコアはリセット直前の 7500 (高値) であること
        assert state.final_scores[0] == 7500
        # game_idx が 1 に進んでいること
        assert state.game_idx == 1
        # ゲーム1 の暫定スコアは 50 で記録されていること
        assert state.final_scores[1] == 50


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

    def test_phantom_board_emitted_when_guard_off(self) -> None:
        """幻盤面ガード OFF (既定) では従来通り emit される (bit-identical)。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_OJAMA, row_count=BOARD_ROWS)
        assert mod._should_emit(state, board, BoardState.STABLE)

    def test_phantom_board_blocked_when_guard_on(self) -> None:
        """幻盤面ガード ON では全面おじゃま盤面が emit されない。

        非試合画面 (対戦カード紹介・ロビー・順位表) 由来の誤認識盤面を
        学習データに入れないための門番 (2026-08-08)。
        """
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_OJAMA, row_count=BOARD_ROWS)
        assert not mod._should_emit(
            state, board, BoardState.STABLE, exclude_phantom=True,
        )

    def test_normal_board_emitted_when_guard_on(self) -> None:
        """幻盤面ガード ON でも実戦の色ぷよ盤面は emit される (偽陽性なし)。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED, row_count=BOARD_ROWS)
        assert mod._should_emit(
            state, board, BoardState.STABLE, exclude_phantom=True,
        )

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
# --sample-interval 間引きロジックのテスト
# ============================

class TestSampleInterval:
    """collect_lean の sample_interval_sec 引数に関するロジックを検証する。

    動画 I/O は使わず、内部で用いる間引きフレーム数の計算のみを検証する。
    """

    def test_step_default_zero_means_every_frame(self) -> None:
        """sample_interval_sec=0.0 のとき step=1 (全フレーム処理) になること。

        collect_lean 内部の計算式:
          sample_interval_frames = max(1, round(interval * fps)) if interval > 0 else 1
        を直接再現して検証する。
        """
        fps = 30.0
        sample_interval_sec = 0.0
        step = max(1, int(round(sample_interval_sec * fps))) \
            if sample_interval_sec > 0.0 else 1
        assert step == 1

    def test_step_calculation_matches_v2_formula(self) -> None:
        """sample_interval_sec=0.1 のとき collect_indicators_v2 と同じ計算結果になること。

        collect_indicators_v2:406 の計算式と同一であることを確認する。
        """
        fps = 30.0
        sample_interval_sec = 0.1
        # collect_boards_lean の計算式
        step_lean = max(1, int(round(sample_interval_sec * fps))) \
            if sample_interval_sec > 0.0 else 1
        # collect_indicators_v2 の計算式 (sample_interval_sec=0 の場合を除く)
        step_v2 = max(1, int(round(sample_interval_sec * fps)))
        assert step_lean == step_v2
        # 30fps × 0.1s = 3 フレームに 1 回
        assert step_lean == 3

    def test_step_02_at_30fps(self) -> None:
        """sample_interval_sec=0.2, fps=30 のとき step=6 になること。"""
        fps = 30.0
        sample_interval_sec = 0.2
        step = max(1, int(round(sample_interval_sec * fps))) \
            if sample_interval_sec > 0.0 else 1
        assert step == 6

    def test_step_minimum_is_1(self) -> None:
        """極端に小さい interval (例: 0.001) でも step が 1 以上になること。"""
        fps = 30.0
        sample_interval_sec = 0.001  # round(0.03) = 0 → max(1, 0) = 1
        step = max(1, int(round(sample_interval_sec * fps))) \
            if sample_interval_sec > 0.0 else 1
        assert step >= 1

    def test_collect_lean_accepts_sample_interval_kwarg(self) -> None:
        """collect_lean が sample_interval_sec キーワード引数を受け付けること。

        inspect でシグネチャを確認し、後方互換 (既定 0.0) を保証する。
        """
        import inspect
        mod = _import_lean()
        sig = inspect.signature(mod.collect_lean)
        assert "sample_interval_sec" in sig.parameters
        default = sig.parameters["sample_interval_sec"].default
        assert default == 0.0

    def test_won_label_preserved_with_sample_interval(self, tmp_path: Path) -> None:
        """_LeanNpzAccumulator の won ラベル付与は sample_interval に無関係に機能すること。

        間引きの有無に関わらず acc.append → assign_won_labels の動作が変わらない
        ことを確認する (collect_lean 本体は動画依存なので内部ロジックのみ検証)。
        """
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # 1P 側 snapshot 3 件 (フレーム間引きで local_i=0, 3, 6 相当を模倣)
        for frame_idx in [0, 3, 6]:
            acc.append(
                _make_board(COLOR_RED)._grid, "v29", "1P",
                float(frame_idx) / 30.0, 0, frame_idx,
            )
        # 2P 側 snapshot 3 件
        for frame_idx in [0, 3, 6]:
            acc.append(
                _make_board(COLOR_BLUE)._grid, "v29", "2P",
                float(frame_idx) / 30.0, 0, frame_idx,
            )
        # 1P 勝ち
        acc.assign_won_labels({0: {"1P": 8000, "2P": 2000}})
        # 全 1P 盤面が won=1.0 になること
        for i, side in enumerate(acc.sides):
            if side == "1P":
                assert float(acc.wons[i]) == 1.0
            else:
                assert float(acc.wons[i]) == 0.0


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


# ============================
# score 保存・復元テスト (修正1)
# ============================

class TestScoreColumn:
    """修正1: score 列の保存・復元を検証する。"""

    def test_score_saved_in_npz(self, tmp_path: Path) -> None:
        """append した score が npz に int32 で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1, score=4800)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 1.0, 0, 1, score=3200)
        out = tmp_path / "score_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert "score" in data
        assert data["score"].dtype == np.int32
        scores = data["score"].tolist()
        assert scores[0] == 4800
        assert scores[1] == 3200

    def test_score_none_saved_as_minus1(self, tmp_path: Path) -> None:
        """score=None は -1 として保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=None)
        out = tmp_path / "score_none.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == -1

    def test_score_default_none_backward_compat(self, tmp_path: Path) -> None:
        """score 引数を省略した場合 (後方互換: 既存呼び出し) も -1 で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # score 引数なしで呼ぶ (後方互換)
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "score_compat.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == -1

    def test_score_roundtrip_multiple(self, tmp_path: Path) -> None:
        """複数 snapshot の score が往復 (save → load) で一致すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        expected_scores = [100, 500, 2000, -1, 9999]
        raw_scores = [100, 500, 2000, None, 9999]
        for i, s in enumerate(raw_scores):
            acc.append(_make_board()._grid, "v29", "1P", float(i), 0, i, score=s)
        out = tmp_path / "score_roundtrip.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["score"].tolist() == expected_scores

    def test_existing_npz_keys_unchanged(self, tmp_path: Path) -> None:
        """score 列追加後も既存キーが全て維持されること (後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=5000)
        out = tmp_path / "compat_keys.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        # 既存キーが全て存在すること
        for key in ("grids", "video_id", "side", "t_sec", "game_idx", "frame_idx", "won"):
            assert key in data, f"後方互換キー '{key}' が消えた"
        # 新規キーも存在すること
        assert "score" in data


# ============================
# next_pair / dnext_pair 保存テスト (指標① 本命版検証用、2026-07 追加)
# ============================

class TestNextPairColumn:
    """next1_a/next1_b/dnext_a/dnext_b の保存・後方互換を検証する。"""

    def test_next_pair_saved_in_npz(self, tmp_path: Path) -> None:
        """next_pair/dnext_pair を渡すと npz に正しい int8 値で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(
            _make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1,
            next_pair=(COLOR_RED, COLOR_BLUE), dnext_pair=(COLOR_GREEN, COLOR_YELLOW),
        )
        out = tmp_path / "next_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["next1_a"][0]) == COLOR_RED
        assert int(data["next1_b"][0]) == COLOR_BLUE
        assert int(data["dnext_a"][0]) == COLOR_GREEN
        assert int(data["dnext_b"][0]) == COLOR_YELLOW

    def test_next_pair_none_saved_as_unknown(self, tmp_path: Path) -> None:
        """next_pair/dnext_pair=None は NEXT_COLOR_UNKNOWN (-1) として保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "next_none.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        for key in ("next1_a", "next1_b", "dnext_a", "dnext_b"):
            assert int(data[key][0]) == mod.NEXT_COLOR_UNKNOWN

    def test_next_pair_backward_compat_omitted_kwarg(self, tmp_path: Path) -> None:
        """next_pair/dnext_pair 引数を省略した既存呼び出しでも -1 埋めされること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # 既存コード同様、next_pair/dnext_pair を渡さない呼び出し
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=1234)
        out = tmp_path / "next_compat.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == 1234
        for key in ("next1_a", "next1_b", "dnext_a", "dnext_b"):
            assert int(data[key][0]) == mod.NEXT_COLOR_UNKNOWN

    def test_next_pair_dtype_int8(self, tmp_path: Path) -> None:
        """next1_a 等が int8 dtype で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(
            _make_board()._grid, "v29", "1P", 1.0, 0, 1,
            next_pair=(COLOR_RED, COLOR_BLUE),
        )
        out = tmp_path / "next_dtype.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        for key in ("next1_a", "next1_b", "dnext_a", "dnext_b"):
            assert data[key].dtype == np.int8

    def test_process_side_lean_passes_through_next_pair(self, tmp_path: Path) -> None:
        """_process_side_lean が next_pair/dnext_pair を acc.append に伝搬すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            next_pair=(COLOR_RED, COLOR_GREEN), dnext_pair=(COLOR_BLUE, COLOR_YELLOW),
        )
        assert acc.next1_as[0] == COLOR_RED
        assert acc.next1_bs[0] == COLOR_GREEN
        assert acc.dnext_as[0] == COLOR_BLUE
        assert acc.dnext_bs[0] == COLOR_YELLOW

    def test_process_side_lean_default_next_pair_is_unknown(self) -> None:
        """_process_side_lean で next_pair/dnext_pair を渡さない場合 -1 埋めされること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
        )
        assert acc.next1_as[0] == mod.NEXT_COLOR_UNKNOWN
        assert acc.dnext_as[0] == mod.NEXT_COLOR_UNKNOWN

    def test_collect_lean_accepts_capture_next_kwarg(self) -> None:
        """collect_lean が capture_next キーワード引数を受け付け、既定 False であること。"""
        import inspect
        mod = _import_lean()
        sig = inspect.signature(mod.collect_lean)
        assert "capture_next" in sig.parameters
        assert sig.parameters["capture_next"].default is False


# ============================
# 窒息フォールバック判定テスト (修正3)
# ============================

class TestWinnerBySurvival:
    """修正3: _winner_by_survival の窒息フォールバック判定を検証する。"""

    def _make_suffocated_board(self) -> Board:
        """3列目の画面内最上段(row=1, col=2)にぷよを置いた窒息盤面を返す。

        2026-07-22 ルール是正: 窒息セルは隠し段(row0)でなく画面内最上段(row1)。
        _DEATH_ROW=1 に合わせる。
        """
        g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        g[1][2] = COLOR_RED  # 窒息セル (画面内最上段)
        # 下段にも適当にぷよ (count_puyos > 0 にする)
        for col in range(BOARD_COLS):
            g[BOARD_ROWS - 1][col] = COLOR_RED
        return Board.from_list(g)

    def test_1p_suffocated_2p_wins(self) -> None:
        """1P が窒息していれば _winner_by_survival が 2P を返すこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # 1P: 窒息盤面
        acc.append(self._make_suffocated_board()._grid, "v29", "1P", 5.0, 0, 50)
        # 2P: 通常盤面 (窒息なし)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 5.0, 0, 50)
        result = mod._winner_by_survival(acc, 0)
        assert result == "2P"

    def test_2p_suffocated_1p_wins(self) -> None:
        """2P が窒息していれば _winner_by_survival が 1P を返すこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # 1P: 通常盤面
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 5.0, 0, 50)
        # 2P: 窒息盤面
        acc.append(self._make_suffocated_board()._grid, "v29", "2P", 5.0, 0, 50)
        result = mod._winner_by_survival(acc, 0)
        assert result == "1P"

    def test_neither_suffocated_returns_none(self) -> None:
        """両者とも窒息なし → None を返すこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board(COLOR_RED)._grid, "v29", "1P", 5.0, 0, 50)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 5.0, 0, 50)
        result = mod._winner_by_survival(acc, 0)
        assert result is None

    def test_no_snapshot_returns_none(self) -> None:
        """snapshot が存在しない game_idx → None を返すこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # game_idx=99 のスナップショットは追加しない
        result = mod._winner_by_survival(acc, 99)
        assert result is None

    def test_assign_won_fallback_to_survival_when_scores_equal(self) -> None:
        """スコアが同点のとき窒息フォールバックで won が付与されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # 1P: 窒息盤面
        g_suf = self._make_suffocated_board()._grid
        acc.append(g_suf, "v29", "1P", 5.0, 0, 50)
        # 2P: 通常盤面
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 5.0, 0, 50)
        # 両者スコア同一 → スコア判定不能 → 窒息フォールバック
        acc.assign_won_labels({0: {"1P": 4000, "2P": 4000}})
        # 1P が窒息 → 2P 勝ち
        assert float(acc.wons[0]) == 0.0  # 1P 側: 負け
        assert float(acc.wons[1]) == 1.0  # 2P 側: 勝ち (2P 視点でなく 1P 視点なので 1P は負け=0)

    def test_score_primary_over_survival(self) -> None:
        """スコアで判定できる場合は窒息に関わらずスコアが優先されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        # 1P: 窒息盤面だが score が高い
        g_suf = self._make_suffocated_board()._grid
        acc.append(g_suf, "v29", "1P", 5.0, 0, 50)
        acc.append(_make_board(COLOR_BLUE)._grid, "v29", "2P", 5.0, 0, 50)
        # スコアは 1P が高い → スコア判定優先
        acc.assign_won_labels({0: {"1P": 8000, "2P": 2000}})
        assert float(acc.wons[0]) == 1.0  # 1P 勝ち (窒息無視)
        assert float(acc.wons[1]) == 0.0  # 2P 負け


# ============================
# _resolve_sample_interval_frames (2026-07-28 追加)
# collect_indicators_v2 と同仕様のフレーム単位間引き指定。
# 動画デコードは一切行わず、fps を差し替えた純粋関数呼び出しのみで検証する。
# ============================


class TestResolveSampleIntervalFrames:
    """フレーム数指定の優先・後方互換・不正値クランプを検証する。"""

    def test_explicit_frames_takes_priority_over_sec(self) -> None:
        """フレーム数指定が秒指定より優先されること (fps が変わっても結果不変)。"""
        mod = _import_lean()
        for fps in (30.0, 60.0):
            resolved = mod._resolve_sample_interval_frames(
                sample_interval_sec=1.0, fps=fps, sample_interval_frames=8,
            )
            assert resolved == 8, f"fps={fps} でもフレーム数指定 8 が優先されるべき"

    def test_omitted_preserves_legacy_sec_behavior_60fps(self) -> None:
        """フレーム数指定省略時、60fps で秒指定の従来換算結果と完全一致すること。"""
        mod = _import_lean()
        resolved = mod._resolve_sample_interval_frames(
            sample_interval_sec=0.2, fps=60.0, sample_interval_frames=None,
        )
        assert resolved == max(1, int(round(0.2 * 60.0)))
        assert resolved == 12

    def test_omitted_preserves_legacy_sec_behavior_30fps(self) -> None:
        """フレーム数指定省略時、30fps でも秒指定の従来換算結果と完全一致すること。"""
        mod = _import_lean()
        resolved = mod._resolve_sample_interval_frames(
            sample_interval_sec=0.2, fps=30.0, sample_interval_frames=None,
        )
        assert resolved == max(1, int(round(0.2 * 30.0)))
        assert resolved == 6

    def test_zero_sec_defaults_to_one_frame(self) -> None:
        """sample_interval_sec=0.0 (全フレーム指定) は従来通り 1 になること。"""
        mod = _import_lean()
        resolved = mod._resolve_sample_interval_frames(
            sample_interval_sec=0.0, fps=60.0, sample_interval_frames=None,
        )
        assert resolved == 1

    @pytest.mark.parametrize("bad_frames", [0, -1, -100])
    def test_non_positive_frames_clamped_to_one(self, bad_frames: int) -> None:
        """0 以下のフレーム数指定は下限 1 に丸められること。"""
        mod = _import_lean()
        resolved = mod._resolve_sample_interval_frames(
            sample_interval_sec=0.0, fps=60.0, sample_interval_frames=bad_frames,
        )
        assert resolved == mod.MIN_SAMPLE_INTERVAL_FRAMES


# ============================
# chain_trigger_sec 保存テスト (2026-07-29 追加、機能D 検知時刻の記録)
# ============================


class TestChainTriggerSecColumn:
    """chain_trigger_sec の保存・後方互換を検証する。"""

    def test_chain_trigger_sec_saved_in_npz(self, tmp_path: Path) -> None:
        """chain_trigger_sec を渡すと npz に float32 で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(
            _make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1,
            chain_trigger_sec=12.5,
        )
        out = tmp_path / "trigger_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert "chain_trigger_sec" in data
        assert data["chain_trigger_sec"].dtype == np.float32
        assert float(data["chain_trigger_sec"][0]) == pytest.approx(12.5)

    def test_chain_trigger_sec_none_saved_as_nan(self, tmp_path: Path) -> None:
        """chain_trigger_sec=None は NaN (CHAIN_TRIGGER_SEC_UNKNOWN) として保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "trigger_none.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert math.isnan(float(data["chain_trigger_sec"][0]))

    def test_chain_trigger_sec_backward_compat_omitted_kwarg(self, tmp_path: Path) -> None:
        """chain_trigger_sec 引数を省略した既存呼び出しでも NaN 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=1234)
        out = tmp_path / "trigger_compat.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == 1234
        assert math.isnan(float(data["chain_trigger_sec"][0]))

    def test_existing_npz_keys_unchanged_after_trigger_sec_addition(self, tmp_path: Path) -> None:
        """chain_trigger_sec 追加後も既存キーが全て維持されること (後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=5000)
        out = tmp_path / "trigger_compat_keys.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        for key in (
            "grids", "video_id", "side", "t_sec", "game_idx", "frame_idx", "won",
            "score", "next1_a", "next1_b", "dnext_a", "dnext_b",
        ):
            assert key in data, f"後方互換キー '{key}' が消えた"
        assert "chain_trigger_sec" in data

    def test_process_side_lean_extracts_trigger_sec_from_chain_event(self) -> None:
        """_process_side_lean が chain_event.trigger_sec を acc.append に伝搬すること。"""
        mod = _import_lean()

        class _FakeChainEvent:
            trigger_sec = 42.5

        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            chain_event=_FakeChainEvent(),
        )
        assert acc.chain_trigger_secs[0] == pytest.approx(42.5)

    def test_process_side_lean_default_trigger_sec_is_nan(self) -> None:
        """chain_event を渡さない場合 (既定 None) は NaN 埋めされること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
        )
        assert math.isnan(acc.chain_trigger_secs[0])

    def test_process_side_lean_chain_event_none_trigger_sec(self) -> None:
        """chain_event=None を明示指定した場合も NaN 埋めされること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            chain_event=None,
        )
        assert math.isnan(acc.chain_trigger_secs[0])


def test_collect_lean_signature_has_sample_interval_frames_appended_at_tail() -> None:
    """collect_lean() の新引数 sample_interval_frames / enable_chain_tracker /

    normalize_fps_30 / enable_effect_gate / effect_gate_persist_sec /
    enable_effect_visual_gate / enable_burst_guard_v2 /
    enable_transition_merge_guard / burst_gate_open_threshold /
    enable_hidden_row_burst_guard / enable_burst_close_extension /
    burst_chain_gap_max_sec / enable_online_hsv_refresh /
    enable_match_transition_debounce /
    enable_ojama_entry_gravity_settle_guard /
    enable_gravity_settle_reset_on_exit が末尾に順次 optional 追加され、
    既存引数の並び・デフォルト値が一切変わっていないこと (backwards compat)。
    """
    import inspect
    mod = _import_lean()
    sig = inspect.signature(mod.collect_lean)
    params = list(sig.parameters.keys())
    assert params[:6] == [
        "video_path", "out_npz", "max_sec", "start_sec",
        "sample_interval_sec", "capture_next",
    ]
    assert params[6] == "sample_interval_frames"
    assert sig.parameters["sample_interval_frames"].default is None
    assert params[7] == "enable_chain_tracker"
    assert sig.parameters["enable_chain_tracker"].default is False
    assert params[8] == "normalize_fps_30"
    # 2026-07-30 既定 True 化 (user承認済み、A/B実測で60fps stride-2が優位)
    assert sig.parameters["normalize_fps_30"].default is True
    # エフェクト時間ゲート (2026-08-03、A/B 計測用): 末尾に追加、既定 OFF。
    assert params[-14] == "enable_effect_gate"
    assert sig.parameters["enable_effect_gate"].default is False
    assert params[-13] == "effect_gate_persist_sec"
    assert sig.parameters["effect_gate_persist_sec"].default is None
    # 案B 4条件AND拡張 (2026-08-04、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-12] == "enable_effect_visual_gate"
    assert sig.parameters["enable_effect_visual_gate"].default is False
    # バーストガード再設計 Stage1 (2026-08-05、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-11] == "enable_burst_guard_v2"
    assert sig.parameters["enable_burst_guard_v2"].default is False
    # バーストガード Stage1.5 (2026-08-05 アーキ追補、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-10] == "enable_transition_merge_guard"
    assert sig.parameters["enable_transition_merge_guard"].default is False
    # バーストガード緊急較正 (2026-08-05、factorialバックテスト用): さらに末尾に追加、既定 None。
    assert params[-9] == "burst_gate_open_threshold"
    assert sig.parameters["burst_gate_open_threshold"].default is None
    # バーストガード Stage1.5b (2026-08-05 アーキ追補、§11、A/B 計測用):
    # さらに末尾に追加、既定 OFF。
    assert params[-8] == "enable_hidden_row_burst_guard"
    assert sig.parameters["enable_hidden_row_burst_guard"].default is False
    # バーストガード §12 close側再設計 (2026-08-05 アーキ確定、A/B 計測用):
    # さらに末尾に追加、既定 OFF。
    assert params[-7] == "enable_burst_close_extension"
    assert sig.parameters["enable_burst_close_extension"].default is False
    # バーストガード §12 緊急パラメータ化 (2026-08-05、A/B 計測用):
    # さらに末尾に追加、既定 None。
    assert params[-6] == "burst_chain_gap_max_sec"
    assert sig.parameters["burst_chain_gap_max_sec"].default is None
    # 長時間劣化修正 A+B (2026-08-06、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-5] == "enable_online_hsv_refresh"
    assert sig.parameters["enable_online_hsv_refresh"].default is False
    # 長時間劣化修正 A' (2026-08-06、§4追補、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-4] == "enable_match_transition_debounce"
    assert sig.parameters["enable_match_transition_debounce"].default is False
    # 状態機械振動バグ B+C 修正 (2026-08-08、A/B 計測用):
    # さらに末尾に追加、既定 OFF (両 OFF で bit-identical)。
    assert params[-3] == "enable_ojama_entry_gravity_settle_guard"
    assert (
        sig.parameters["enable_ojama_entry_gravity_settle_guard"].default is False
    )
    assert params[-2] == "enable_gravity_settle_reset_on_exit"
    assert sig.parameters["enable_gravity_settle_reset_on_exit"].default is False
    # 幻盤面ガード (2026-08-08、非試合画面の除外): さらに末尾に追加、既定 OFF。
    assert params[-1] == "enable_phantom_board_guard"
    assert sig.parameters["enable_phantom_board_guard"].default is False
    assert sig.parameters["enable_burst_guard_v2"].default is False


def test_collect_lean_enable_chain_tracker_default_false_backward_compat() -> None:
    """enable_chain_tracker 省略時は False (= 従来通り VideoChainTracker 無効)。

    既存呼び出し (引数省略) の挙動を一切変えないことを保証する
    (2026-07-30 全フレーム基準データ収集の CHAIN 凍結欠陥修正で追加)。
    """
    import inspect
    mod = _import_lean()
    sig = inspect.signature(mod.collect_lean)
    assert sig.parameters["enable_chain_tracker"].default is False


def test_main_cli_has_enable_chain_tracker_flag_default_false() -> None:
    """CLI --enable-chain-tracker フラグが store_true・既定 False で追加されている。"""
    import argparse
    from unittest.mock import patch

    mod = _import_lean()
    captured: dict[str, object] = {}

    def _fake_collect_lean(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    with patch.object(mod, "collect_lean", _fake_collect_lean):
        with patch(
            "sys.argv",
            ["collect_boards_lean.py", "--video", "x.mp4", "--out-npz", "y.npz"],
        ):
            mod.main()
    assert captured["enable_chain_tracker"] is False


def _run_fake_main_lean(argv_tail: list[str]) -> dict[str, object]:
    """collect_lean を差し替えて main() を実行し、渡された kwargs を返す共通ヘルパ。"""
    from unittest.mock import patch

    mod = _import_lean()
    captured: dict[str, object] = {}

    def _fake_collect_lean(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    with patch.object(mod, "collect_lean", _fake_collect_lean):
        with patch(
            "sys.argv",
            ["collect_boards_lean.py", "--video", "x.mp4", "--out-npz", "y.npz"]
            + argv_tail,
        ):
            mod.main()
    return captured


def test_main_cli_enable_hidden_row_burst_guard_default_false() -> None:
    """CLI で --enable-hidden-row-burst-guard 未指定なら False が渡ること。

    バーストガード Stage1.5b (2026-08-05 アーキ追補、§11、backwards compat)。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_hidden_row_burst_guard"] is False


def test_main_cli_enable_hidden_row_burst_guard_flag_sets_true() -> None:
    """--enable-hidden-row-burst-guard 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-hidden-row-burst-guard"])
    assert captured["enable_hidden_row_burst_guard"] is True


def test_main_cli_enable_burst_close_extension_default_false() -> None:
    """CLI で --enable-burst-close-extension 未指定なら False が渡ること。

    バーストガード §12 close側再設計 (2026-08-05 アーキ確定、backwards compat)。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_burst_close_extension"] is False


def test_main_cli_enable_burst_close_extension_flag_sets_true() -> None:
    """--enable-burst-close-extension 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-burst-close-extension"])
    assert captured["enable_burst_close_extension"] is True


def test_main_cli_burst_chain_gap_max_sec_default_none() -> None:
    """CLI で --burst-chain-gap-max 未指定なら None が渡ること (§12 緊急パラメータ化)。"""
    captured = _run_fake_main_lean([])
    assert captured["burst_chain_gap_max_sec"] is None


def test_main_cli_burst_chain_gap_max_sec_zero_disables_extension() -> None:
    """--burst-chain-gap-max 0 指定時は 0.0 が渡ること (延長無効化用途)。"""
    captured = _run_fake_main_lean(["--burst-chain-gap-max", "0"])
    assert captured["burst_chain_gap_max_sec"] == 0.0


def test_main_cli_enable_online_hsv_refresh_default_false() -> None:
    """CLI で --enable-online-hsv-refresh 未指定なら False が渡ること (長時間劣化修正A+B)。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_online_hsv_refresh"] is False


def test_main_cli_enable_online_hsv_refresh_flag_sets_true() -> None:
    """--enable-online-hsv-refresh 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-online-hsv-refresh"])
    assert captured["enable_online_hsv_refresh"] is True


def test_main_cli_enable_match_transition_debounce_default_false() -> None:
    """CLI で --enable-match-transition-debounce 未指定なら False が渡ること

    (長時間劣化修正A'、2026-08-06、§4追補)。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_match_transition_debounce"] is False


def test_main_cli_enable_match_transition_debounce_flag_sets_true() -> None:
    """--enable-match-transition-debounce 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-match-transition-debounce"])
    assert captured["enable_match_transition_debounce"] is True


def test_main_cli_normalize_fps_30_default_true_when_no_flags() -> None:
    """CLI で --normalize-fps-30 / --no-normalize-fps-30 とも未指定なら

    normalize_fps_30=True が collect_lean に渡る (2026-07-30 既定 True 化)。
    """
    captured = _run_fake_main_lean([])
    assert captured["normalize_fps_30"] is True


def test_main_cli_no_normalize_fps_30_flag_disables() -> None:
    """--no-normalize-fps-30 指定時は normalize_fps_30=False が渡ること。"""
    captured = _run_fake_main_lean(["--no-normalize-fps-30"])
    assert captured["normalize_fps_30"] is False


def test_main_cli_no_normalize_fps_30_wins_when_both_specified() -> None:
    """--normalize-fps-30 と --no-normalize-fps-30 を同時指定した場合、

    無効化 (--no-normalize-fps-30) が優先されること (coordinator指示の仕様)。
    """
    captured = _run_fake_main_lean(["--normalize-fps-30", "--no-normalize-fps-30"])
    assert captured["normalize_fps_30"] is False


# ============================
# collect_lean() ループ結合テスト (2026-07-30 追加)
# 実動画は使わず cv2.VideoCapture / RecognitionPipeline.load_default を
# フェイクに差し替え、normalize_fps_30 の stride 自動注入・優先順位・
# 既定OFF bit-identical を直接検証する (test_collect_indicators_v2.py の
# _FakeCapture パターンを流用)。
# ============================


class _FakeCaptureLean:
    """cv2.VideoCapture の最小フェイク。固定本数のダミーフレームを返す。"""

    def __init__(self, n_frames: int, fps: float = 30.0) -> None:
        self._n_frames = n_frames
        self._fps = fps
        self._i = 0
        mod = _import_lean()
        self._frame = np.zeros((mod.TARGET_H, mod.TARGET_W, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._n_frames)
        return 0.0

    def set(self, prop: int, value: float) -> None:  # noqa: D401 - フェイクなので no-op
        pass

    def read(self) -> "tuple[bool, np.ndarray | None]":
        if self._i >= self._n_frames:
            return False, None
        self._i += 1
        return True, self._frame

    def release(self) -> None:
        pass


class _FakeLeanPipeline:
    """RecognitionPipeline.load_default の最小フェイク (collect_lean 用)。"""

    def __init__(self) -> None:
        self.update_calls: list[int] = []

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        self.update_calls.append(fi)
        board = Board.from_list([[0] * BOARD_COLS for _ in range(BOARD_ROWS)])
        side = SimpleNamespace(
            state=BoardState.MENU,  # STABLE でないため acc.append は呼ばれない
            score=None, confirmed_board=board,
            next_pair=None, dnext_pair=None, chain_event=None,
        )
        return SimpleNamespace(p1=side, p2=side)

    def set_video_id(self, video_id: str) -> None:
        pass


def _run_fake_collect_lean(
    tmp_path: Path, n_frames: int, *, fps: float = 30.0, **collect_kwargs: object,
) -> "tuple[int, _FakeLeanPipeline]":
    """cv2.VideoCapture / RecognitionPipeline.load_default をフェイクに
    差し替えて collect_lean() を実行する共通ヘルパ。
    """
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(n_frames, fps=fps)
    fake_pipeline = _FakeLeanPipeline()

    def _fake_video_capture(_path: str) -> _FakeCaptureLean:
        return fake_cap

    def _fake_load_default(*args: object, **kwargs: object) -> _FakeLeanPipeline:
        return fake_pipeline

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", _fake_video_capture)
        mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
        out_npz = tmp_path / "out.npz"
        n = mod.collect_lean(Path("dummy_video.mp4"), out_npz, **collect_kwargs)
    return n, fake_pipeline


def test_collect_lean_normalize_fps_30_default_omitted_applies_stride_2_for_60fps(
    tmp_path: Path,
) -> None:
    """normalize_fps_30 省略時 (2026-07-30 既定 True 化後) は 60fps 動画に

    stride-2 が自動適用され、2フレームに1回だけ認識されること。
    """
    n_frames = 10
    _, fake_pipeline = _run_fake_collect_lean(tmp_path, n_frames, fps=60.0)
    assert fake_pipeline.update_calls == list(range(0, n_frames, 2))


def test_collect_lean_normalize_fps_30_explicit_false_is_bit_identical(
    tmp_path: Path,
) -> None:
    """normalize_fps_30=False を明示指定した場合は 60fps でも間引かれず

    全フレーム認識される (CLI --no-normalize-fps-30 相当、後方互換経路の保持)。
    """
    n_frames = 10
    _, fake_pipeline = _run_fake_collect_lean(
        tmp_path, n_frames, fps=60.0, normalize_fps_30=False,
    )
    assert len(fake_pipeline.update_calls) == n_frames


def test_collect_lean_normalize_fps_30_60fps_injects_stride_2(
    tmp_path: Path,
) -> None:
    """normalize_fps_30=True かつ 60fps のとき、2フレームに1回だけ認識される

    (resolve_normalize_fps_30_stride(60.0) == 2 の自動注入を直接確認)。
    """
    n_frames = 10
    _, fake_pipeline = _run_fake_collect_lean(
        tmp_path, n_frames, fps=60.0, normalize_fps_30=True,
    )
    assert fake_pipeline.update_calls == list(range(0, n_frames, 2))


def test_collect_lean_normalize_fps_30_ignored_when_sample_interval_frames_explicit(
    tmp_path: Path,
) -> None:
    """明示 sample_interval_frames が normalize_fps_30 より優先されること。"""
    n_frames = 12
    _, fake_pipeline = _run_fake_collect_lean(
        tmp_path, n_frames, fps=60.0,
        sample_interval_frames=4, normalize_fps_30=True,
    )
    assert fake_pipeline.update_calls == list(range(0, n_frames, 4))


# ============================
# 共有ゲーム境界カウンタ (2026-07-31 desync 根治)
# ============================

class TestSharedGameCounter:
    """1P/2P 共有の game_idx が desync を防ぐことを検証する。

    旧実装は _SideState.game_idx を side ごとに独立して進めていたため:
      - 検知フレームがずれると game_idx がずれる (実測 57.6% が 5秒超)
      - 片側の score OCR が壊れている動画 (c26/c58 等) ではその side が
        game 0 に留まり、以降すべてのゲームが対応しなくなる
    """

    def test_both_sides_share_index_when_one_side_detects(self) -> None:
        """片側だけが境界を検知しても両者の game_idx が揃う。

        これが旧実装の最大の欠陥 (score OCR が片側で壊れている動画)。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter()
        s1, s2 = mod._SideState(), mod._SideState()
        # 1P はスコアが進んでリセットする / 2P はスコアを読めない (None)
        mod._update_game_boundary(s1, 5000, shared=shared, t_sec=10.0)
        mod._update_game_boundary(s2, None, shared=shared, t_sec=10.0)
        mod._update_game_boundary(s1, 100, shared=shared, t_sec=60.0)  # 境界
        assert shared.game_idx == 1
        assert s1.game_idx == 1
        # 2P はこの後 score を読めた時点で共有カウンタに追従する
        mod._update_game_boundary(s2, 300, shared=shared, t_sec=61.0)
        assert s2.game_idx == 1, "片側検知の境界に追従していない"

    def test_same_boundary_detected_by_both_advances_once(self) -> None:
        """両者が同じ境界を検知しても game_idx は 1 だけ進む (デバウンス)。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter()
        s1, s2 = mod._SideState(), mod._SideState()
        mod._update_game_boundary(s1, 8000, shared=shared, t_sec=10.0)
        mod._update_game_boundary(s2, 7000, shared=shared, t_sec=10.0)
        # 同一境界を 1P が t=60.0、2P が t=60.5 で検知
        mod._update_game_boundary(s1, 0, shared=shared, t_sec=60.0)
        mod._update_game_boundary(s2, 0, shared=shared, t_sec=60.5)
        assert shared.game_idx == 1, "同一境界で 2 回進んでいる"
        assert s1.game_idx == s2.game_idx == 1

    def test_separate_boundaries_beyond_debounce_advance_twice(self) -> None:
        """デバウンス幅を超えて離れた 2 つの境界は別ゲームとして数える。

        1 試合は最短 14 秒なので、デバウンス 5 秒は実試合を潰さない。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter()
        s1 = mod._SideState()
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=10.0)
        mod._update_game_boundary(s1, 0, shared=shared, t_sec=60.0)   # 境界1
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=90.0)
        mod._update_game_boundary(s1, 0, shared=shared, t_sec=120.0)  # 境界2
        assert shared.game_idx == 2

    def test_final_scores_recorded_under_shared_index(self) -> None:
        """final_scores が共有 game_idx をキーに記録される。

        ここがずれると _merge_final_scores が別ゲーム同士を突き合わせる
        (勝敗ラベルが壊れる真因)。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter()
        s1, s2 = mod._SideState(), mod._SideState()
        for t, sc1, sc2 in ((10.0, 1000, 2000), (20.0, 5000, 6000)):
            mod._update_game_boundary(s1, sc1, shared=shared, t_sec=t)
            mod._update_game_boundary(s2, sc2, shared=shared, t_sec=t)
        # 1P だけが境界を検知 (2P は score が読めない)
        mod._update_game_boundary(s1, 0, shared=shared, t_sec=60.0)
        mod._update_game_boundary(s2, 100, shared=shared, t_sec=60.0)
        # game 0 の最終スコアは両者ともリセット直前の高値
        assert s1.final_scores[0] == 5000
        assert s2.final_scores[0] == 6000
        # game 1 は両者とも同じキーに入る
        assert 1 in s1.final_scores and 1 in s2.final_scores

    def test_shared_none_keeps_legacy_independent_behaviour(self) -> None:
        """shared=None なら従来の side 独立カウンタ (後方互換)。"""
        mod = _import_lean()
        s1 = mod._SideState()
        mod._update_game_boundary(s1, 5000)
        mod._update_game_boundary(s1, 0)
        assert s1.game_idx == 1
