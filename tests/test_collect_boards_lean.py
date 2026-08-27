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
from src.ojama_accounting import OjamaAccountSnapshot
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

    def test_raw_pixel_stable_default_true_bit_identical(self) -> None:
        """raw_pixel_stable 省略時 (既定 True) は従来挙動と bit-identical。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        assert mod._should_emit(state, board, BoardState.STABLE)

    def test_raw_pixel_stable_false_no_longer_blocks_emit(self) -> None:
        """(2026-08-18 二次追加、STABLE持続ゲートの役割転用):
        raw_pixel_stable=False を渡しても emit は拒否されない。

        旧仕様は記録拒否 (61%収集減 + 局面偏りが実測で判明したため撤回)。
        判定結果は npz の stable_persistence_confidence 列にタグとして残す
        方式に変わった (_process_side_lean 参照)。raw_pixel_stable 引数
        自体は後方互換のため signature に残すが _should_emit 内では無視する。
        """
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        assert mod._should_emit(
            state, board, BoardState.STABLE, raw_pixel_stable=False,
        )

    def test_raw_pixel_stable_true_allows_emit(self) -> None:
        """raw_pixel_stable=True を明示指定しても従来通り emit される。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        assert mod._should_emit(
            state, board, BoardState.STABLE, raw_pixel_stable=True,
        )


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


# ============================
# tsumo_count 保存テスト (2026-08-12 追加、おじゃま収支近似復元 v3 の
# 着地イベントゲート用。dedup済み STABLE snapshot は1着地に対応しないため、
# RecognitionPipeline.tsumo_count(side) の増分を着地イベントの代理指標に使う)
# ============================


class TestTsumoCountColumn:
    """tsumo_count の保存・後方互換を検証する。"""

    def test_tsumo_count_saved_in_npz(self, tmp_path: Path) -> None:
        """tsumo_count を渡すと npz に int32 で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(
            _make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1,
            tsumo_count=7,
        )
        out = tmp_path / "tsumo_count_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert "tsumo_count" in data
        assert data["tsumo_count"].dtype == np.int32
        assert int(data["tsumo_count"][0]) == 7

    def test_tsumo_count_none_saved_as_unknown(self, tmp_path: Path) -> None:
        """tsumo_count=None は TSUMO_COUNT_UNKNOWN (-1) として保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "tsumo_count_none.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["tsumo_count"][0]) == mod.TSUMO_COUNT_UNKNOWN

    def test_tsumo_count_backward_compat_omitted_kwarg(self, tmp_path: Path) -> None:
        """tsumo_count 引数を省略した既存呼び出しでも -1 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=1234)
        out = tmp_path / "tsumo_count_compat.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == 1234
        assert int(data["tsumo_count"][0]) == mod.TSUMO_COUNT_UNKNOWN

    def test_tsumo_count_roundtrip_multiple(self, tmp_path: Path) -> None:
        """複数 snapshot の tsumo_count が往復 (save → load) で一致すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        expected = [0, 3, 5, -1, 12]
        raw = [0, 3, 5, None, 12]
        for i, tc in enumerate(raw):
            acc.append(
                _make_board()._grid, "v29", "1P", float(i), 0, i,
                tsumo_count=tc,
            )
        out = tmp_path / "tsumo_count_roundtrip.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["tsumo_count"].tolist() == expected

    def test_existing_npz_keys_unchanged_after_tsumo_count_addition(
        self, tmp_path: Path,
    ) -> None:
        """tsumo_count 追加後も既存キーが全て維持されること (後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=5000)
        out = tmp_path / "tsumo_count_compat_keys.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        for key in (
            "grids", "video_id", "side", "t_sec", "game_idx", "frame_idx", "won",
            "score", "next1_a", "next1_b", "dnext_a", "dnext_b",
            "chain_trigger_sec",
        ):
            assert key in data, f"後方互換キー '{key}' が消えた"
        assert "tsumo_count" in data

    def test_process_side_lean_passes_through_tsumo_count(self) -> None:
        """_process_side_lean が tsumo_count を acc.append に伝搬すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            tsumo_count=9,
        )
        assert acc.tsumo_counts[0] == 9

    def test_process_side_lean_default_tsumo_count_is_unknown(self) -> None:
        """tsumo_count を渡さない場合 (既定 None) は -1 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
        )
        assert acc.tsumo_counts[0] == mod.TSUMO_COUNT_UNKNOWN

    def test_collect_lean_calls_pipeline_tsumo_count_for_both_sides(
        self, tmp_path: Path,
    ) -> None:
        """collect_lean の主ループが 1P/2P 双方の pipeline.tsumo_count を呼ぶこと。

        RecognitionPipeline.tsumo_count(side) は stateless getter であり、
        _process_side_lean 呼び出し前に main loop で取得して渡す設計
        (2026-08-12 追加)。
        """
        n_frames = 4
        _, fake_pipeline = _run_fake_collect_lean(tmp_path, n_frames, fps=30.0)
        # MENU 状態で emit されないため tsumo_counts 自体は空だが、
        # pipeline.tsumo_count は毎認識フレームで両 side 分呼ばれる。
        assert fake_pipeline.tsumo_count_calls.count("1P") == n_frames
        assert fake_pipeline.tsumo_count_calls.count("2P") == n_frames

    def test_collect_lean_tolerates_pipeline_without_tsumo_count(
        self, tmp_path: Path,
    ) -> None:
        """pipeline が tsumo_count 未対応でも collect_lean は例外を出さないこと
        (後方互換: 古いフェイク/差し替えオブジェクトでも動く)。
        """
        mod = _import_lean()
        fake_cap = _FakeCaptureLean(4, fps=30.0)

        class _NoTsumoCountPipeline(_FakeLeanPipeline):
            """tsumo_count メソッドを持たないフェイク。"""
            def __getattribute__(self, name: str) -> object:
                if name == "tsumo_count":
                    raise AttributeError(name)
                return super().__getattribute__(name)

        fake_pipeline = _NoTsumoCountPipeline()

        def _fake_video_capture(_path: str) -> _FakeCaptureLean:
            return fake_cap

        def _fake_load_default(*args: object, **kwargs: object) -> _NoTsumoCountPipeline:
            return fake_pipeline

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod.cv2, "VideoCapture", _fake_video_capture)
            mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
            out_npz = tmp_path / "no_tsumo_count.npz"
            n = mod.collect_lean(Path("dummy_video.mp4"), out_npz)
        assert n == 0  # MENU 状態のため snapshot は 0 件だが例外なく完了


# ============================
# all_clear_pending 保存テスト (2026-08-12 追加、全消しボーナス予約中フラグ。
# post-hoc の score 跳ね検出近似は過検出気味 (c143実測 ON率6.7%) だったため、
# src.chain_detector.VideoChainTracker.all_clear_pending が実運用パイプラインで
# 厳密に追跡済みの値をそのまま npz へ記録する)
# ============================


class TestAllClearPendingColumn:
    """all_clear_pending の保存・後方互換を検証する。"""

    def test_all_clear_pending_saved_in_npz(self, tmp_path: Path) -> None:
        """all_clear_pending を渡すと npz に int8 で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(
            _make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1,
            all_clear_pending=1,
        )
        out = tmp_path / "all_clear_pending_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert "all_clear_pending" in data
        assert data["all_clear_pending"].dtype == np.int8
        assert int(data["all_clear_pending"][0]) == 1

    def test_all_clear_pending_none_saved_as_unknown(self, tmp_path: Path) -> None:
        """all_clear_pending=None は ALL_CLEAR_PENDING_UNKNOWN (-1) として
        保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "all_clear_pending_none.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["all_clear_pending"][0]) == mod.ALL_CLEAR_PENDING_UNKNOWN

    def test_all_clear_pending_backward_compat_omitted_kwarg(
        self, tmp_path: Path,
    ) -> None:
        """all_clear_pending 引数を省略した既存呼び出しでも -1 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=1234)
        out = tmp_path / "all_clear_pending_compat.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == 1234
        assert int(data["all_clear_pending"][0]) == mod.ALL_CLEAR_PENDING_UNKNOWN

    def test_all_clear_pending_roundtrip_multiple(self, tmp_path: Path) -> None:
        """複数 snapshot の all_clear_pending が往復 (save → load) で
        一致すること (bool / int / None を混在させる)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        expected = [1, 0, -1, 1, 0]
        raw = [True, False, None, 1, 0]
        for i, v in enumerate(raw):
            acc.append(
                _make_board()._grid, "v29", "1P", float(i), 0, i,
                all_clear_pending=v,
            )
        out = tmp_path / "all_clear_pending_roundtrip.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert data["all_clear_pending"].tolist() == expected

    def test_existing_npz_keys_unchanged_after_all_clear_pending_addition(
        self, tmp_path: Path,
    ) -> None:
        """all_clear_pending 追加後も既存キーが全て維持されること (後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=5000)
        out = tmp_path / "all_clear_pending_compat_keys.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        for key in (
            "grids", "video_id", "side", "t_sec", "game_idx", "frame_idx", "won",
            "score", "next1_a", "next1_b", "dnext_a", "dnext_b",
            "chain_trigger_sec", "tsumo_count",
        ):
            assert key in data, f"後方互換キー '{key}' が消えた"
        assert "all_clear_pending" in data

    def test_process_side_lean_passes_through_all_clear_pending(self) -> None:
        """_process_side_lean が all_clear_pending を acc.append に伝搬すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            all_clear_pending=1,
        )
        assert acc.all_clear_pendings[0] == 1

    def test_process_side_lean_default_all_clear_pending_is_unknown(self) -> None:
        """all_clear_pending を渡さない場合 (既定 None) は -1 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
        )
        assert acc.all_clear_pendings[0] == mod.ALL_CLEAR_PENDING_UNKNOWN

    def test_collect_lean_all_clear_pending_flows_from_chain_tracker_to_npz(
        self, tmp_path: Path,
    ) -> None:
        """collect_lean の主ループが pipeline の side別 VideoChainTracker
        (_chain_tracker_1p / _chain_tracker_2p) の all_clear_pending を読み取り、
        npz まで正しく伝搬すること (2026-08-12 追加)。
        """
        mod = _import_lean()
        fake_cap = _FakeCaptureLean(1, fps=30.0)
        fake_pipeline = _FakeLeanPipelineStableAllClear(
            pending_1p=True, pending_2p=False,
        )

        def _fake_video_capture(_path: str) -> _FakeCaptureLean:
            return fake_cap

        def _fake_load_default(
            *args: object, **kwargs: object,
        ) -> "_FakeLeanPipelineStableAllClear":
            return fake_pipeline

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod.cv2, "VideoCapture", _fake_video_capture)
            mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
            out_npz = tmp_path / "all_clear_pending_flow.npz"
            n = mod.collect_lean(Path("dummy_video.mp4"), out_npz)
        assert n == 2  # 1P/2P それぞれ 1 snapshot
        data = np.load(str(out_npz), allow_pickle=True)
        by_side = dict(zip(data["side"].tolist(), data["all_clear_pending"].tolist()))
        assert by_side["1P"] == 1
        assert by_side["2P"] == 0

    def test_collect_lean_tolerates_pipeline_without_chain_tracker_attrs(
        self, tmp_path: Path,
    ) -> None:
        """pipeline が _chain_tracker_1p / _chain_tracker_2p を持たなくても
        collect_lean は例外を出さないこと (後方互換: 古いフェイク/差し替え
        オブジェクトでも動く、2026-08-12 追加)。
        """
        mod = _import_lean()
        fake_cap = _FakeCaptureLean(4, fps=30.0)
        fake_pipeline = _FakeLeanPipeline()  # _chain_tracker_1p/2p 属性なし

        def _fake_video_capture(_path: str) -> _FakeCaptureLean:
            return fake_cap

        def _fake_load_default(*args: object, **kwargs: object) -> _FakeLeanPipeline:
            return fake_pipeline

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod.cv2, "VideoCapture", _fake_video_capture)
            mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
            out_npz = tmp_path / "no_chain_tracker.npz"
            n = mod.collect_lean(Path("dummy_video.mp4"), out_npz)
        assert n == 0  # MENU 状態のため snapshot は 0 件だが例外なく完了


# ============================
# ojama_net_balance / ojama_forecast 保存テスト (2026-08-12 追加。
# npz からの事後復元が不可能と確定したため (score近似v1/v2は相関0.33-0.38で
# 不合格、tsumo_countゲートv3も不可判定)、収集中に OjamaAccountingTracker を
# 実際に駆動して真値を記録する。own-perspective (自分視点) の値)
# ============================


class TestOjamaAccountingColumns:
    """ojama_net_balance / ojama_forecast の保存・後方互換・駆動を検証する。"""

    def test_ojama_columns_saved_in_npz(self, tmp_path: Path) -> None:
        """値を渡すと npz に float32 で保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(
            _make_board(COLOR_RED)._grid, "v29", "1P", 1.0, 0, 1,
            ojama_net_balance=5.0, ojama_forecast=3.0,
        )
        out = tmp_path / "ojama_columns_test.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert "ojama_net_balance" in data
        assert "ojama_forecast" in data
        assert data["ojama_net_balance"].dtype == np.float32
        assert data["ojama_forecast"].dtype == np.float32
        assert float(data["ojama_net_balance"][0]) == pytest.approx(5.0)
        assert float(data["ojama_forecast"][0]) == pytest.approx(3.0)

    def test_ojama_columns_none_saved_as_nan(self, tmp_path: Path) -> None:
        """ojama_net_balance/ojama_forecast=None は NaN として保存されること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1)
        out = tmp_path / "ojama_columns_none.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert math.isnan(float(data["ojama_net_balance"][0]))
        assert math.isnan(float(data["ojama_forecast"][0]))

    def test_ojama_columns_backward_compat_omitted_kwarg(
        self, tmp_path: Path,
    ) -> None:
        """ojama_* 引数を省略した既存呼び出しでも NaN 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=1234)
        out = tmp_path / "ojama_columns_compat.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        assert int(data["score"][0]) == 1234
        assert math.isnan(float(data["ojama_net_balance"][0]))
        assert math.isnan(float(data["ojama_forecast"][0]))

    def test_existing_npz_keys_unchanged_after_ojama_columns_addition(
        self, tmp_path: Path,
    ) -> None:
        """ojama_* 追加後も既存キーが全て維持されること (後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board()._grid, "v29", "1P", 1.0, 0, 1, score=5000)
        out = tmp_path / "ojama_columns_compat_keys.npz"
        acc.save(out)
        data = np.load(str(out), allow_pickle=True)
        for key in (
            "grids", "video_id", "side", "t_sec", "game_idx", "frame_idx", "won",
            "score", "next1_a", "next1_b", "dnext_a", "dnext_b",
            "chain_trigger_sec", "tsumo_count", "all_clear_pending",
        ):
            assert key in data, f"後方互換キー '{key}' が消えた"
        assert "ojama_net_balance" in data
        assert "ojama_forecast" in data

    def test_process_side_lean_passes_through_ojama_columns(self) -> None:
        """_process_side_lean が ojama_net_balance/ojama_forecast を
        acc.append に伝搬すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            ojama_net_balance=2.5, ojama_forecast=1.5,
        )
        assert acc.ojama_net_balances[0] == pytest.approx(2.5)
        assert acc.ojama_forecasts[0] == pytest.approx(1.5)

    def test_process_side_lean_default_ojama_columns_is_nan(self) -> None:
        """ojama_* を渡さない場合 (既定 None) は NaN 埋めされること
        (後方互換: 既存呼び出しコードは一切変更不要)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
        )
        assert math.isnan(acc.ojama_net_balances[0])
        assert math.isnan(acc.ojama_forecasts[0])

    def test_drive_ojama_accounting_lean_moves_net_balance_on_chain_fire(
        self,
    ) -> None:
        """1P が連鎖を撃つと own-perspective net_balance/forecast が正しく
        動くこと (物理妥当性チェック: score跳躍→2P forecast 増加→net反転)。

        src/ojama_accounting.py の on_state_transition 駆動シーケンス
        (tests/test_ojama_accounting.py の _fire_chain と同じ組み立て) を
        _drive_ojama_accounting_lean 経由で再現する。
        """
        from src.ojama_accounting import K_SETTLE_FRAMES
        mod = _import_lean()
        tracker = mod.OjamaAccountingTracker()
        tracker.reset()
        state_p1 = mod._SideState()
        state_p2 = mod._SideState()
        prev = {"p1": BoardState.STABLE, "p2": BoardState.STABLE}

        def _drive(
            state1: BoardState, score1: int, state2: BoardState, score2: int,
            t: float,
        ) -> OjamaAccountSnapshot:
            p1 = SimpleNamespace(state=state1, score=score1)
            p2 = SimpleNamespace(state=state2, score=score2)
            snap = mod._drive_ojama_accounting_lean(
                tracker, state_p1, state_p2, prev["p1"], prev["p2"],
                p1, p2, None, None, t,
            )
            prev["p1"], prev["p2"] = state1, state2
            return snap

        # 1P 連鎖開始 (350点 = 5個お邪魔 // 70) → 終了 → settle 確定
        _drive(BoardState.CHAIN, 0, BoardState.STABLE, 0, 5.0)
        snap = _drive(BoardState.STABLE, 350, BoardState.STABLE, 0, 7.0)
        for i in range(K_SETTLE_FRAMES):
            snap = _drive(
                BoardState.STABLE, 350, BoardState.STABLE, 0,
                7.0 + (i + 1) * 0.001,
            )

        net_1p, fc_1p, net_2p, fc_2p = mod._ojama_snapshot_to_own_perspective(
            snap,
        )
        assert fc_2p == pytest.approx(5.0)  # 2P に 5個の予告が来ている
        assert fc_1p == pytest.approx(0.0)
        assert net_1p > 0.0  # 1P 有利方向
        assert net_2p == pytest.approx(-net_1p)  # 符号反転で own-perspective

    def test_collect_lean_ojama_columns_finite_for_stable_snapshot(
        self, tmp_path: Path,
    ) -> None:
        """collect_lean の主ループが ojama_net_balance/ojama_forecast を
        npz まで伝搬し、有限値 (非NaN) で own-perspective 符号反転が
        成立すること (2026-08-12 追加)。
        """
        mod = _import_lean()
        fake_cap = _FakeCaptureLean(1, fps=30.0)
        fake_pipeline = _FakeLeanPipelineStableAllClear(
            pending_1p=False, pending_2p=False,
        )

        def _fake_video_capture(_path: str) -> _FakeCaptureLean:
            return fake_cap

        def _fake_load_default(
            *args: object, **kwargs: object,
        ) -> "_FakeLeanPipelineStableAllClear":
            return fake_pipeline

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod.cv2, "VideoCapture", _fake_video_capture)
            mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
            out_npz = tmp_path / "ojama_columns_flow.npz"
            n = mod.collect_lean(Path("dummy_video.mp4"), out_npz)
        assert n == 2  # 1P/2P それぞれ 1 snapshot
        data = np.load(str(out_npz), allow_pickle=True)
        assert "ojama_net_balance" in data
        assert "ojama_forecast" in data
        by_side_net = dict(
            zip(data["side"].tolist(), data["ojama_net_balance"].tolist()),
        )
        # 1 フレームのみ・連鎖なしのため net は 0 かつ 1P/2P で符号反転一致
        assert not math.isnan(by_side_net["1P"])
        assert not math.isnan(by_side_net["2P"])
        assert by_side_net["1P"] == pytest.approx(-by_side_net["2P"])

    def test_drain_ojama_by_tsumo_delta_lean_does_not_call_pipeline(
        self,
    ) -> None:
        """_drain_ojama_by_tsumo_delta_lean は pipeline.tsumo_count() を
        再度呼ばず、渡された値のみを使うこと (main loop の呼び出し回数を
        変えないための設計、2026-08-12 追加)。tsumo_count=None なら何もしない。
        """
        mod = _import_lean()
        tracker = mod.OjamaAccountingTracker()
        tracker.reset()
        state = mod._SideState()
        # None のときは drain しない (呼び出し不能でも例外にならない)
        mod._drain_ojama_by_tsumo_delta_lean(tracker, "p1", state, None, 1.0)
        assert state.ojama_prev_tsumo == 0
        # 増分ありのときは on_tsumo_settled が delta 回呼ばれ prev_tsumo が更新される
        mod._drain_ojama_by_tsumo_delta_lean(tracker, "p1", state, 3, 2.0)
        assert state.ojama_prev_tsumo == 3


def test_collect_lean_signature_has_sample_interval_frames_appended_at_tail() -> None:
    """collect_lean() の新引数 sample_interval_frames / enable_chain_tracker /

    normalize_fps_30 / enable_effect_gate / effect_gate_persist_sec /
    enable_effect_visual_gate / enable_burst_guard_v2 /
    enable_transition_merge_guard / burst_gate_open_threshold /
    enable_hidden_row_burst_guard / enable_burst_close_extension /
    burst_chain_gap_max_sec / enable_online_hsv_refresh /
    enable_match_transition_debounce /
    enable_ojama_entry_gravity_settle_guard /
    enable_gravity_settle_reset_on_exit / enable_phantom_board_guard /
    enable_margin_time_rate / enable_stable_majority_window /
    enable_ojama_fall_placement_override / enable_ojama_fall_entry_hardening /
    enable_chain_gate_raw_fallback / enable_ojama_fall_scoped_exit /
    precise_seek / enable_highlight_override / enable_patch_fp_hsv_guard /
    enable_boundary_multisignal / enable_winner_panel_crosscheck /
    enable_floating_gap_restore / enable_landing_color_guard /
    enable_stable_persistence_gate / enable_match_end_persist_override /
    enable_post_match_lockdown_latch / enable_result_screen_hardening /
    enable_ojama_fall_color_swap_guard / enable_chain_formula_read_verify /
    enable_formula_chain_count_update / enable_formula_step_interlude
    が末尾に順次 optional 追加され、既存引数の並び・デフォルト値
    が一切変わっていないこと (backwards compat)。
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
    # 注記 (2026-08-17 発見・W23根治タスクで是正、W25根治タスクでさらに
    # -1シフト、2026-08-18 (d) STABLE持続確認タスクでさらに -1シフト、
    # 連鎖中物理推論配線タスクでさらに -1シフト、盤面収集作り替え
    # (1手区切りスケジューラ+持続的物理制約フィルタ) タスクでさらに
    # -2シフト、W26根治タスクでさらに -1シフト、STABLE凍結デッドロック根治
    # 3フラグ配線タスク (2026-08-24) でさらに -3シフト):
    # 以下のインデックスは enable_override_color_guard /
    # enable_ojama_column_stack_fix / enable_next_history_starvation_fix /
    # enable_ojama_cnn_override_warmup / enable_stable_persistence_gate /
    # enable_chain_estimate_recording / enable_move_segmented_recording /
    # enable_physics_persistence_filter / enable_ojama_fall_color_swap_guard /
    # enable_chain_formula_read_verify / enable_formula_chain_count_update /
    # enable_formula_step_interlude の 12件が末尾にさらに追加された分、
    # 元の値から一律 -12 シフトしてある (旧値は git log 参照)。
    assert params[-50] == "enable_effect_gate"
    assert sig.parameters["enable_effect_gate"].default is False
    assert params[-49] == "effect_gate_persist_sec"
    assert sig.parameters["effect_gate_persist_sec"].default is None
    # 案B 4条件AND拡張 (2026-08-04、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-48] == "enable_effect_visual_gate"
    assert sig.parameters["enable_effect_visual_gate"].default is False
    # バーストガード再設計 Stage1 (2026-08-05、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-47] == "enable_burst_guard_v2"
    assert sig.parameters["enable_burst_guard_v2"].default is False
    # バーストガード Stage1.5 (2026-08-05 アーキ追補、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-46] == "enable_transition_merge_guard"
    assert sig.parameters["enable_transition_merge_guard"].default is False
    # バーストガード緊急較正 (2026-08-05、factorialバックテスト用): さらに末尾に追加、既定 None。
    assert params[-45] == "burst_gate_open_threshold"
    assert sig.parameters["burst_gate_open_threshold"].default is None
    # バーストガード Stage1.5b (2026-08-05 アーキ追補、§11、A/B 計測用):
    # さらに末尾に追加、既定 OFF。
    assert params[-44] == "enable_hidden_row_burst_guard"
    assert sig.parameters["enable_hidden_row_burst_guard"].default is False
    # バーストガード §12 close側再設計 (2026-08-05 アーキ確定、A/B 計測用):
    # さらに末尾に追加、既定 OFF。
    assert params[-43] == "enable_burst_close_extension"
    assert sig.parameters["enable_burst_close_extension"].default is False
    # バーストガード §12 緊急パラメータ化 (2026-08-05、A/B 計測用):
    # さらに末尾に追加、既定 None。
    assert params[-42] == "burst_chain_gap_max_sec"
    assert sig.parameters["burst_chain_gap_max_sec"].default is None
    # 長時間劣化修正 A+B (2026-08-06、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-41] == "enable_online_hsv_refresh"
    assert sig.parameters["enable_online_hsv_refresh"].default is False
    # 長時間劣化修正 A' (2026-08-06、§4追補、A/B 計測用): さらに末尾に追加、既定 OFF。
    assert params[-40] == "enable_match_transition_debounce"
    assert sig.parameters["enable_match_transition_debounce"].default is False
    # 状態機械振動バグ B+C 修正 (2026-08-08、A/B 計測用):
    # さらに末尾に追加、既定 OFF (両 OFF で bit-identical)。
    assert params[-39] == "enable_ojama_entry_gravity_settle_guard"
    assert (
        sig.parameters["enable_ojama_entry_gravity_settle_guard"].default is False
    )
    assert params[-38] == "enable_gravity_settle_reset_on_exit"
    assert sig.parameters["enable_gravity_settle_reset_on_exit"].default is False
    # 幻盤面ガード (2026-08-08、非試合画面の除外): さらに末尾に追加、既定 OFF。
    assert params[-37] == "enable_phantom_board_guard"
    assert sig.parameters["enable_phantom_board_guard"].default is False
    # マージンタイム逓減 (2026-08-09): さらに末尾に追加、既定 OFF。
    assert params[-36] == "enable_margin_time_rate"
    assert sig.parameters["enable_margin_time_rate"].default is False
    # 盤面確定窓 3中2多数決 (2026-08-13 user承認): さらに末尾に追加、既定 OFF。
    assert params[-35] == "enable_stable_majority_window"
    assert sig.parameters["enable_stable_majority_window"].default is False
    # OJAMA_FALL誤分類根因調査 案2/案4-lite/案3 (2026-08-13):
    # さらに末尾に追加、既定 OFF (全 OFF で bit-identical)。
    assert params[-34] == "enable_ojama_fall_placement_override"
    assert (
        sig.parameters["enable_ojama_fall_placement_override"].default is False
    )
    assert params[-33] == "enable_ojama_fall_entry_hardening"
    assert sig.parameters["enable_ojama_fall_entry_hardening"].default is False
    assert params[-32] == "enable_chain_gate_raw_fallback"
    assert sig.parameters["enable_chain_gate_raw_fallback"].default is False
    # OJAMA_FALL出口の根治 案1 (2026-08-13、フル物差し回帰タスク#5向け新設):
    # さらに末尾に追加、既定 OFF (bit-identical)。collect_boards_lean.py には
    # 元々 RecognitionPipeline 側の実装のみ存在し CLI 未配線だったギャップの
    # 是正 (config (c) = OJAMA_FALL系3種の物差し比較に必要)。
    assert params[-31] == "enable_ojama_fall_scoped_exit"
    assert sig.parameters["enable_ojama_fall_scoped_exit"].default is False
    # フレーム精度シーク (2026-08-14、タスク#5 物差し回帰で発見した測定器
    # 事故の修正): さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-30] == "precise_seek"
    assert sig.parameters["precise_seek"].default is False
    # W13根治 案1 (2026-08-16): highlight override 配線。さらに末尾に追加、
    # 既定 OFF (bit-identical、物差しv2 A/B測定用)。
    assert params[-29] == "enable_highlight_override"
    assert sig.parameters["enable_highlight_override"].default is False
    # W13根治 案2 (2026-08-17): tier1 patch-NCC HSV AND ガード配線。
    # さらに末尾に追加、既定 OFF (bit-identical、案1 との A/B/併用測定用)。
    assert params[-28] == "enable_patch_fp_hsv_guard"
    assert sig.parameters["enable_patch_fp_hsv_guard"].default is False
    # W20/W21根治 (2026-08-17): 試合境界マルチシグナル配線。さらに末尾に
    # 追加、既定 OFF (bit-identical)。
    assert params[-27] == "enable_boundary_multisignal"
    assert sig.parameters["enable_boundary_multisignal"].default is False
    # W20/W21根治 (2026-08-17): 勝者パネルクロスチェック配線。さらに末尾に
    # 追加、既定 OFF (bit-identical)。
    assert params[-26] == "enable_winner_panel_crosscheck"
    assert sig.parameters["enable_winner_panel_crosscheck"].default is False
    # R2 浮きぷよ是正機構 (2026-08-17): さらに末尾に追加、既定 OFF
    # (bit-identical、hsv-guard 併用/単独 A/B 測定用)。
    assert params[-25] == "enable_floating_gap_restore"
    assert sig.parameters["enable_floating_gap_restore"].default is False
    # W10根治 (2026-08-17): 着地セル色の継続監視ガード CLI 配線漏れの是正
    # (認識強化統一測定タスクで発見)。さらに末尾に追加、既定 OFF
    # (bit-identical)。
    assert params[-24] == "enable_landing_color_guard"
    assert sig.parameters["enable_landing_color_guard"].default is False
    # 持続誤認26件系統1/2 (2026-08-17、docs/KNOWN_WEAKNESSES.md W10):
    # さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-23] == "enable_override_color_guard"
    assert sig.parameters["enable_override_color_guard"].default is False
    assert params[-22] == "enable_ojama_column_stack_fix"
    assert sig.parameters["enable_ojama_column_stack_fix"].default is False
    # W23根治 (2026-08-17、docs/KNOWN_WEAKNESSES.md W23): _validate_next_history
    # の ever_seen 飢餓状態対策。さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-21] == "enable_next_history_starvation_fix"
    assert sig.parameters["enable_next_history_starvation_fix"].default is False
    # W25根治 案4 (2026-08-17、docs/KNOWN_WEAKNESSES.md W25): おじゃま落下
    # 白雲パーティクル誤認対策。さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-20] == "enable_ojama_cnn_override_warmup"
    assert sig.parameters["enable_ojama_cnn_override_warmup"].default is False
    # W25根治 第3弾・最終 (2026-08-18、docs/KNOWN_WEAKNESSES.md W25):
    # CNN観測入力段の会計整合フィルタ。さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-19] == "enable_ojama_write_accounting_guard"
    assert sig.parameters["enable_ojama_write_accounting_guard"].default is False
    # (d) STABLE持続確認 (2026-08-18、docs/BOUNDARY_MULTISIGNAL_DESIGN_
    # 2026-08-17.md §5): さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-18] == "enable_stable_persistence_gate"
    assert sig.parameters["enable_stable_persistence_gate"].default is False
    # (b-1) match_end持続時間ゲート (2026-08-18): さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-17] == "enable_match_end_persist_override"
    assert sig.parameters["enable_match_end_persist_override"].default is False
    # (b-2) 次試合開始までのラッチ (2026-08-18): さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-16] == "enable_post_match_lockdown_latch"
    assert sig.parameters["enable_post_match_lockdown_latch"].default is False
    # 境界実装の仕上げ (enable_result_screen_hardening、2026-08-18): さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-15] == "enable_result_screen_hardening"
    assert sig.parameters["enable_result_screen_hardening"].default is False
    # 連鎖中物理推論の配線 (enable_chain_estimate_recording、2026-08-18):
    # さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-14] == "enable_chain_estimate_recording"
    assert sig.parameters["enable_chain_estimate_recording"].default is False
    # 1手区切り観測スケジューラ (enable_move_segmented_recording、
    # 2026-08-18、盤面収集の作り替え本体): さらに末尾に追加、既定 OFF
    # (bit-identical)。
    assert params[-13] == "enable_move_segmented_recording"
    assert sig.parameters["enable_move_segmented_recording"].default is False
    # 持続的物理制約フィルタ (enable_physics_persistence_filter、
    # 2026-08-18、盤面収集の作り替え本体): さらに末尾に追加、既定 OFF
    # (bit-identical)。
    assert params[-12] == "enable_physics_persistence_filter"
    assert sig.parameters["enable_physics_persistence_filter"].default is False
    # W26根治 (enable_ojama_fall_color_swap_guard、2026-08-18、
    # docs/KNOWN_WEAKNESSES.md W26節): さらに末尾に追加、既定 OFF
    # (bit-identical)。
    assert params[-11] == "enable_ojama_fall_color_swap_guard"
    assert sig.parameters["enable_ojama_fall_color_swap_guard"].default is False
    # (b-2)ラッチ解除の数値スコア化 + 補助解除 (2026-08-19、user指示「必ず
    # 試合前スコアは0」): さらに末尾に追加、既定 OFF (bit-identical)。
    assert params[-10] == "enable_lockdown_score_numeric_release"
    assert (
        sig.parameters["enable_lockdown_score_numeric_release"].default is False
    )
    assert params[-9] == "enable_lockdown_score_moving_release"
    assert (
        sig.parameters["enable_lockdown_score_moving_release"].default is False
    )
    # MatchEndDetector NCC 閾値上書き (2026-08-19、全消しテロップ誤検出対策の
    # A/B 用): さらに末尾に追加、既定 None (bit-identical)。
    assert params[-8] == "match_end_ncc_threshold"
    assert sig.parameters["match_end_ncc_threshold"].default is None
    # 新試合証拠ゲート (2026-08-19、偽境界の断片化対策): さらに末尾に追加、
    # 既定 OFF (bit-identical)。
    assert params[-7] == "enable_boundary_newmatch_evidence"
    # 試合境界の 0 リセット要求 (2026-08-20、user 指摘「減るのはただの誤認」):
    # さらに末尾へ追加、既定 OFF。試合中の score は単調増加しかしないため、
    # 減少を境界の根拠にしない (連鎖中の1桁誤読による偽境界を根絶する)。
    assert params[-6] == "enable_score_reset_requires_zero"
    # 勝者判定のパネル優先 (2026-08-20、user 決定「パネル優先でいいです」):
    # さらに末尾へ追加、既定 OFF。得点系統の「高い方が勝ち」は約98%しか
    # 成立しないため、明確に読めたパネルを優先する。
    assert params[-5] == "enable_winner_panel_priority"
    # ネイティブ (Rust) HSV セル分類 (2026-08-20): さらに末尾へ追加、既定 OFF。
    # 認識結果は bit-identical で、実測 1 frame 34.69→29.05ms (1.19倍)。
    assert params[-4] == "enable_native_hsv_classifier"
    assert sig.parameters["enable_native_hsv_classifier"].default is False
    # STABLE 凍結デッドロック根治 3 フラグ (2026-08-24、RECOGNITION_ADOPTED
    # 採用、user 承認): さらに末尾へ追加、既定 OFF (bit-identical)。
    assert params[-3] == "enable_chain_formula_read_verify"
    assert sig.parameters["enable_chain_formula_read_verify"].default is False
    assert params[-2] == "enable_formula_chain_count_update"
    assert sig.parameters["enable_formula_chain_count_update"].default is False
    assert params[-1] == "enable_formula_step_interlude"
    assert sig.parameters["enable_formula_step_interlude"].default is False
    assert sig.parameters["enable_winner_panel_priority"].default is False
    assert sig.parameters["enable_score_reset_requires_zero"].default is False
    assert (
        sig.parameters["enable_boundary_newmatch_evidence"].default is False
    )



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


def test_main_cli_enable_highlight_override_default_false() -> None:
    """CLI で --enable-highlight-override 未指定なら False が渡ること。

    W13根治 案1 (2026-08-16、物差しv2 A/B測定用): backwards compat、
    既定 False で従来挙動と bit-identical。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_highlight_override"] is False


def test_main_cli_enable_highlight_override_flag_sets_true() -> None:
    """--enable-highlight-override 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-highlight-override"])
    assert captured["enable_highlight_override"] is True


def test_main_cli_enable_patch_fp_hsv_guard_default_false() -> None:
    """CLI で --enable-patch-fp-hsv-guard 未指定なら False が渡ること。

    W13根治 案2 (2026-08-17、物差しv2 A/B/併用測定用): backwards compat、
    既定 False で従来挙動と bit-identical。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_patch_fp_hsv_guard"] is False


def test_main_cli_enable_patch_fp_hsv_guard_flag_sets_true() -> None:
    """--enable-patch-fp-hsv-guard 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-patch-fp-hsv-guard"])
    assert captured["enable_patch_fp_hsv_guard"] is True


def test_main_cli_enable_floating_gap_restore_default_false() -> None:
    """CLI で --enable-floating-gap-restore 未指定なら False が渡ること。

    R2 浮きぷよ是正機構 (2026-08-17、hsv-guard 併用/単独 A/B 測定用):
    backwards compat、既定 False で従来挙動と bit-identical。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_floating_gap_restore"] is False


def test_main_cli_enable_floating_gap_restore_flag_sets_true() -> None:
    """--enable-floating-gap-restore 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-floating-gap-restore"])
    assert captured["enable_floating_gap_restore"] is True


def test_main_cli_enable_landing_color_guard_default_false() -> None:
    """CLI で --enable-landing-color-guard 未指定なら False が渡ること。

    W10根治 (2026-08-17、認識強化統一測定タスクで発見した CLI 配線漏れの
    是正): backwards compat、既定 False で従来挙動と bit-identical。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_landing_color_guard"] is False


def test_main_cli_enable_landing_color_guard_flag_sets_true() -> None:
    """--enable-landing-color-guard 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-landing-color-guard"])
    assert captured["enable_landing_color_guard"] is True


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
        # tsumo_count(side) の呼び出し記録 (2026-08-12 追加、配線確認用)。
        self.tsumo_count_calls: list[str] = []

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

    def tsumo_count(self, side: str) -> int:
        """RecognitionPipeline.tsumo_count(side) のフェイク実装。

        呼び出しを記録するだけで固定値 0 を返す (2026-08-12 追加)。
        """
        self.tsumo_count_calls.append(side)
        return 0


class _FakeChainTrackerAllClear:
    """VideoChainTracker の all_clear_pending 部分だけを模擬する最小フェイク
    (2026-08-12 追加、collect_lean の main loop 配線確認用)。"""

    def __init__(self, pending: bool) -> None:
        self._pending = pending

    @property
    def all_clear_pending(self) -> bool:
        return self._pending


class _FakeLeanPipelineStableAllClear(_FakeLeanPipeline):
    """side別 VideoChainTracker (_chain_tracker_1p/2p) を保持し、1 フレームだけ
    STABLE を返すフェイク (2026-08-12 追加)。all_clear_pending が npz まで
    伝搬することを検証するための専用フェイク。
    """

    def __init__(self, *, pending_1p: bool, pending_2p: bool) -> None:
        super().__init__()
        self._chain_tracker_1p = _FakeChainTrackerAllClear(pending_1p)
        self._chain_tracker_2p = _FakeChainTrackerAllClear(pending_2p)

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        self.update_calls.append(fi)
        board = _make_board(COLOR_RED)
        side_1p = SimpleNamespace(
            state=BoardState.STABLE, score=100, confirmed_board=board,
            next_pair=None, dnext_pair=None, chain_event=None,
        )
        side_2p = SimpleNamespace(
            state=BoardState.STABLE, score=200, confirmed_board=board,
            next_pair=None, dnext_pair=None, chain_event=None,
        )
        return SimpleNamespace(p1=side_1p, p2=side_2p)


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
# precise_seek (2026-08-14、タスク#5 物差し回帰で発見した測定器事故の修正):
# cv2/ffmpeg の CAP_PROP_POS_FRAMES シークは GOP 構造依存で不正確な場合が
# あり、再エンコード世代が異なる動画で --start-sec 窓の絶対フレーム番号が
# 実内容とずれる (2026-08-14 実測: YouTube再DL動画の人手ラベル突合が
# 52-61%まで崩壊、再DLしていない動画は97%超を維持)。precise_seek=True は
# cap.set() を廃し frame 0 から read() で読み捨てる厳密シークに切り替える。
# ============================


def test_collect_lean_precise_seek_default_false_uses_cap_set(
    tmp_path: Path,
) -> None:
    """precise_seek 省略時 (既定 False) は従来通り cap.set() のみで、

    読み捨てループは実行されない (fake の set() は no-op のため、read() は
    ちょうど end_frame-start_frame 回だけ呼ばれる、backwards compat)。
    """
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(n_frames=10, fps=10.0)
    fake_pipeline = _FakeLeanPipeline()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **k: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        mod.collect_lean(
            Path("dummy_video.mp4"), out_npz,
            start_sec=0.3, max_sec=0.0, normalize_fps_30=False,
        )
    # start_frame = int(0.3 * 10.0) = 3。読み捨てなしなので fake は
    # 合計 (10 - 3) = 7 回だけ read() される (cap._i == 7)。
    assert fake_cap._i == 7
    assert fake_pipeline.update_calls == [3, 4, 5, 6, 7, 8, 9]


def test_collect_lean_precise_seek_true_discards_frames_via_read(
    tmp_path: Path,
) -> None:
    """precise_seek=True では cap.set() の代わりに read() で start_frame 回

    読み捨ててから本処理に入るため、fake の合計 read() 回数が
    (読み捨て3回 + 本処理7回) = 10 回になる (全フレーム消費、backwards
    compat のため update_calls の fi 値自体は False 時と同じ)。
    """
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(n_frames=10, fps=10.0)
    fake_pipeline = _FakeLeanPipeline()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **k: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        mod.collect_lean(
            Path("dummy_video.mp4"), out_npz,
            start_sec=0.3, max_sec=0.0, normalize_fps_30=False,
            precise_seek=True,
        )
    # 読み捨て3回 (discard) + 本処理7回 (update_calls) = fake 全10フレーム消費。
    assert fake_cap._i == 10
    assert fake_pipeline.update_calls == [3, 4, 5, 6, 7, 8, 9]


def test_collect_lean_precise_seek_default_omitted_kwarg_backward_compat(
    tmp_path: Path,
) -> None:
    """precise_seek を省略した既存呼び出し (kwarg 未指定) が

    既定 False と bit-identical であること (backwards compat の直接確認)。
    """
    n_frames = 10
    _, fake_pipeline_omitted = _run_fake_collect_lean(
        tmp_path, n_frames, fps=10.0, start_sec=0.3, max_sec=0.0,
        normalize_fps_30=False,
    )
    _, fake_pipeline_explicit = _run_fake_collect_lean(
        tmp_path, n_frames, fps=10.0, start_sec=0.3, max_sec=0.0,
        normalize_fps_30=False, precise_seek=False,
    )
    assert fake_pipeline_omitted.update_calls == fake_pipeline_explicit.update_calls


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


# ============================
# 試合境界マルチシグナル (W20/W21根治、2026-08-17 追加)
# ============================

class TestBoundaryMultisignal:
    """_SharedGameCounter.observe_visual_signal / multisignal_mode の検証。

    設計: is_match_active (視覚+score_zero+match_end_locked の統合判定) の
    False→True 立ち上がりを境界進行の主信号にし、score-reset は
    フォールバック + 異常マークに降格する (docs/KNOWN_WEAKNESSES.md W20/W21)。
    """

    def test_multisignal_mode_false_is_noop_bit_identical(self) -> None:
        """multisignal_mode=False (既定) では observe_visual_signal は
        game_idx に一切影響しない (旧 score-reset 単独動作と bit-identical)。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter()
        assert shared.multisignal_mode is False
        shared.observe_visual_signal(False, t_sec=1.0)
        shared.observe_visual_signal(True, t_sec=2.0)  # False→True だが無視
        assert shared.game_idx == 0
        assert shared.last_visual_advance_sec is None

    def test_visual_rising_edge_advances_boundary(self) -> None:
        """multisignal_mode=True で False→True 立ち上がりが、
        BOUNDARY_VISUAL_RISE_PERSIST_SEC 秒持続確認後に境界を進める
        (W22根治、2026-08-17: ノイズ除去のため即時確定ではなくなった)。
        確定時刻は持続確認完了時刻ではなく立ち上がり本来の時刻 (2.0) を使う。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        shared.observe_visual_signal(False, t_sec=1.0)
        shared.observe_visual_signal(True, t_sec=2.0)  # 立ち上がり候補
        assert shared.game_idx == 0, "持続確認が完了するまでは進まない"
        shared.observe_visual_signal(
            True, t_sec=2.0 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
        )  # 持続確認完了 → 境界確定
        assert shared.game_idx == 1
        assert shared.last_visual_advance_sec == 2.0
        assert shared.last_visual_rise_sec == 2.0

    def test_visual_rise_flicker_below_persist_is_rejected_as_noise(self) -> None:
        """持続時間が BOUNDARY_VISUAL_RISE_PERSIST_SEC 未満で False に戻る
        瞬き (フェード暗転・ロゴ明滅相当) は境界を進めない (W22根治)。
        瞬きの直後に来る本物の立ち上がりは、瞬きに引きずられず独立に
        持続確認されることも併せて確認する。
        """
        mod = _import_lean()
        persist = mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC
        shared = mod._SharedGameCounter(multisignal_mode=True)
        shared.observe_visual_signal(False, t_sec=1.0)
        shared.observe_visual_signal(True, t_sec=2.0)  # 瞬き立ち上がり (候補1)
        flicker_end_sec = 2.0 + persist * 0.1  # 持続確認完了前に消える
        shared.observe_visual_signal(False, t_sec=flicker_end_sec)  # ノイズ破棄
        assert shared.game_idx == 0, "ノイズ候補で進んでしまっている"
        real_rise_sec = flicker_end_sec  # 直後に本物の立ち上がり (候補2)
        shared.observe_visual_signal(True, t_sec=real_rise_sec)
        shared.observe_visual_signal(
            True, t_sec=real_rise_sec + persist,
        )  # 持続確認完了
        assert shared.game_idx == 1
        assert shared.last_visual_rise_sec == real_rise_sec

    def test_visual_signal_continuous_true_does_not_readvance(self) -> None:
        """True が継続する間は再度進まない (立ち上がりのみ検知)。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        shared.observe_visual_signal(False, t_sec=1.0)
        shared.observe_visual_signal(True, t_sec=2.0)
        shared.observe_visual_signal(True, t_sec=3.0)
        shared.observe_visual_signal(True, t_sec=4.0)
        assert shared.game_idx == 1

    def test_score_reset_near_visual_advance_no_anomaly(self) -> None:
        """score-reset が視覚信号確認済み時刻の近傍なら異常マークしない。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        s1 = mod._SideState()
        shared.observe_visual_signal(False, t_sec=59.0)
        shared.observe_visual_signal(True, t_sec=60.0)  # 視覚信号で境界1 候補
        shared.observe_visual_signal(
            True, t_sec=60.0 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
        )  # 持続確認完了
        assert shared.game_idx == 1
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=10.0)
        # score-reset 検知が視覚信号 (60.0) の許容窓 (3秒) 以内
        mod._update_game_boundary(
            s1, 0, shared=shared, t_sec=61.0, side_label="1P",
        )
        assert shared.game_idx == 1, "視覚信号確認済みで二重に進んでいる"
        assert shared.anomalies == []

    def test_score_reset_without_visual_signal_records_anomaly_and_advances(
        self,
    ) -> None:
        """score-reset が視覚信号なしで発生 → フォールバックで進みつつ
        異常マークが残る (人手レビュー対象、W21実例 c109 相当のケース)。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        s1 = mod._SideState()
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=10.0)
        # 視覚信号は一度も立ち上がっていない状態で score だけ急落
        mod._update_game_boundary(
            s1, 0, shared=shared, t_sec=60.0, side_label="1P",
        )
        assert shared.game_idx == 1, "フォールバックで進んでいない"
        assert len(shared.anomalies) == 1
        anomaly = shared.anomalies[0]
        assert anomaly["side"] == "1P"
        assert anomaly["score_delta"] == 9000
        assert anomaly["t_sec"] == 60.0

    def test_score_reset_far_from_visual_advance_records_anomaly(self) -> None:
        """視覚信号はあったが score-reset 時刻から許容窓より遠い → 異常マーク。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        s1 = mod._SideState()
        shared.observe_visual_signal(False, t_sec=9.0)
        shared.observe_visual_signal(True, t_sec=10.0)  # 視覚信号で境界1候補 (別ゲーム)
        shared.observe_visual_signal(
            True, t_sec=10.0 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
        )  # 持続確認完了
        assert shared.game_idx == 1
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=20.0)
        # 許容窓 3秒より遠い (視覚信号 last_visual_rise_sec=10.0 との差 40 秒)
        mod._update_game_boundary(
            s1, 0, shared=shared, t_sec=50.0, side_label="1P",
        )
        assert shared.game_idx == 2, "遠い score-reset がフォールバックで進んでいない"
        assert len(shared.anomalies) == 1

    def test_score_reset_precedes_visual_rise_race_online_flags_anomaly(
        self,
    ) -> None:
        """W22実例の競合ケース: score-reset が視覚信号の立ち上がりより
        時系列で先に処理される (実測 c109 の常態パターン)。

        単一フレーム順パスでは score-reset の時点で視覚信号はまだ確定
        していないため、オンライン判定は異常マークする (救済は動画処理
        完了後の _reconcile_boundary_anomalies が担う、別テストで検証)。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        s1 = mod._SideState()
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=9.97)
        # score-reset がまず先着 (t=10.0)。視覚信号はまだ立ち上がっていない。
        mod._update_game_boundary(
            s1, 0, shared=shared, t_sec=10.0, side_label="1P",
        )
        assert shared.game_idx == 1, "フォールバックで進んでいない"
        assert len(shared.anomalies) == 1, "score 先着はオンラインでは異常マークされる"
        # 視覚信号は score-reset の 0.03〜3秒後に遅れて到着し、
        # 持続確認を経て確定する (実測範囲を模す)。
        shared.observe_visual_signal(False, t_sec=10.02)
        shared.observe_visual_signal(True, t_sec=10.1)  # 立ち上がり候補
        shared.observe_visual_signal(
            True, t_sec=10.1 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
        )  # 持続確認完了 (advance_if_new はデバウンス内で失敗するが
        # last_visual_rise_sec は無条件で記録される、W22根治の核心)
        assert shared.game_idx == 1, "advance が二重に進んではいけない"
        assert shared.last_visual_rise_sec == 10.1
        assert shared.visual_rise_times == [10.1]
        # ここまではまだ異常マークが残っている (事後救済前)
        assert len(shared.anomalies) == 1

    def test_reconcile_boundary_anomalies_removes_race_false_positive(
        self,
    ) -> None:
        """_reconcile_boundary_anomalies が動画処理完了後に visual_rise_times
        と再突合し、score 先着による偽陽性の異常マークを取り除く
        (W22根治の本丸)。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        s1 = mod._SideState()
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=9.97)
        mod._update_game_boundary(
            s1, 0, shared=shared, t_sec=10.0, side_label="1P",
        )
        assert len(shared.anomalies) == 1
        shared.observe_visual_signal(False, t_sec=10.02)
        shared.observe_visual_signal(True, t_sec=10.1)
        shared.observe_visual_signal(
            True, t_sec=10.1 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
        )
        mod._reconcile_boundary_anomalies(shared)
        assert shared.anomalies == [], (
            "視覚信号が許容窓内に確認できたので事後救済で消えるはず"
        )

    def test_reconcile_boundary_anomalies_keeps_true_positive(self) -> None:
        """視覚信号が許容窓の外にしか無い真の異常は事後突合でも残る
        (救済ロジックが何でも消してしまわないことの回帰防止)。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        s1 = mod._SideState()
        mod._update_game_boundary(s1, 9000, shared=shared, t_sec=9.97)
        mod._update_game_boundary(
            s1, 0, shared=shared, t_sec=10.0, side_label="1P",
        )
        assert len(shared.anomalies) == 1
        # 視覚信号は遠い時刻 (許容窓 3秒 を大きく超える 100 秒後) にのみ存在
        shared.observe_visual_signal(False, t_sec=109.9)
        shared.observe_visual_signal(True, t_sec=110.0)
        shared.observe_visual_signal(
            True, t_sec=110.0 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
        )
        mod._reconcile_boundary_anomalies(shared)
        assert len(shared.anomalies) == 1, "遠い視覚信号で誤って救済してしまっている"


# ============================
# 勝敗演出ロックダウン区間マーク (W20/W21根治、2026-08-17 追加)
# ============================

class TestMatchEndLockedColumn:
    """_LeanNpzAccumulator の match_end_locked 列の記録・保存を検証する。"""

    def test_append_default_omitted_uses_unknown_sentinel(
        self, tmp_path: Path,
    ) -> None:
        """match_end_locked 省略時は MATCH_END_LOCKED_UNKNOWN (-1、後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 10)
        assert acc.match_end_lockeds == [mod.MATCH_END_LOCKED_UNKNOWN]
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert data["match_end_locked"][0] == mod.MATCH_END_LOCKED_UNKNOWN

    def test_append_explicit_values_roundtrip(self, tmp_path: Path) -> None:
        """match_end_locked=True/False が npz に 1/0 でそのまま保存される。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(
            board._grid, "v1", "1P", 1.0, 0, 10, match_end_locked=True,
        )
        acc.append(
            board._grid, "v1", "1P", 2.0, 0, 11, match_end_locked=False,
        )
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert list(data["match_end_locked"]) == [1, 0]


# ============================
# 次試合開始までのラッチ活性フラグ列 (境界実装の仕上げ、2026-08-18 追加)
# ============================

class TestPostMatchLockdownActiveColumn:
    """_LeanNpzAccumulator の post_match_lockdown_active 列の記録・保存を
    検証する (match_end_lockeds と同じマーカー列方式)。"""

    def test_append_default_omitted_uses_unknown_sentinel(
        self, tmp_path: Path,
    ) -> None:
        """省略時は POST_MATCH_LOCKDOWN_ACTIVE_UNKNOWN (-1、後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 10)
        assert acc.post_match_lockdown_actives == [
            mod.POST_MATCH_LOCKDOWN_ACTIVE_UNKNOWN
        ]
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert (
                data["post_match_lockdown_active"][0]
                == mod.POST_MATCH_LOCKDOWN_ACTIVE_UNKNOWN
            )

    def test_append_explicit_values_roundtrip(self, tmp_path: Path) -> None:
        """True/False が npz に 1/0 でそのまま保存される。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(
            board._grid, "v1", "1P", 1.0, 0, 10,
            post_match_lockdown_active=True,
        )
        acc.append(
            board._grid, "v1", "1P", 2.0, 0, 11,
            post_match_lockdown_active=False,
        )
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert list(data["post_match_lockdown_active"]) == [1, 0]

    def test_existing_match_end_locked_column_unaffected(
        self, tmp_path: Path,
    ) -> None:
        """新列の追加が既存 match_end_locked 列の挙動を変えないこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(
            board._grid, "v1", "1P", 1.0, 0, 10,
            match_end_locked=True, post_match_lockdown_active=False,
        )
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert data["match_end_locked"][0] == 1
            assert data["post_match_lockdown_active"][0] == 0


class _FakeLeanPipelinePostMatchLockdown(_FakeLeanPipeline):
    """pipeline._post_match_lockdown_active を保持する最小フェイク
    (2026-08-18 追加、main loop の getattr 配線をフル経路で確認する用)。
    1 フレームだけ STABLE を返し、他は MENU (dedup/空盤面ガードを回避)。
    """

    def __init__(self, lockdown_active: bool) -> None:
        super().__init__()
        self._post_match_lockdown_active = lockdown_active

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        self.update_calls.append(fi)
        board = _make_board(COLOR_RED) if fi == 0 else Board.from_list(
            [[0] * BOARD_COLS for _ in range(BOARD_ROWS)],
        )
        state = BoardState.STABLE if fi == 0 else BoardState.MENU
        side = SimpleNamespace(
            state=state, score=100, confirmed_board=board,
            next_pair=None, dnext_pair=None, chain_event=None,
        )
        return SimpleNamespace(
            p1=side, p2=side, is_match_active=True, match_end_locked=False,
        )


def test_collect_lean_wires_post_match_lockdown_active_to_npz(
    tmp_path: Path,
) -> None:
    """main loop が pipeline._post_match_lockdown_active を getattr し、
    npz の post_match_lockdown_active 列にそのまま反映することをフル配線で
    確認する (True の場合)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(2, fps=30.0)
    fake_pipeline = _FakeLeanPipelinePostMatchLockdown(lockdown_active=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        n = mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
        )
    # frame 0 (STABLE) は 1P/2P 両方が emit される (同一 side オブジェクトを
    # 両方に使うフェイクの仕様のため)。
    assert n == 2
    with np.load(out_npz) as data:
        assert list(data["post_match_lockdown_active"]) == [1, 1]


def test_collect_lean_post_match_lockdown_active_unknown_when_absent(
    tmp_path: Path,
) -> None:
    """pipeline が _post_match_lockdown_active 属性を持たない (旧フェイク等)
    場合は POST_MATCH_LOCKDOWN_ACTIVE_UNKNOWN (-1) のまま (後方互換)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(2, fps=30.0)

    class _FakeNoLockdownAttr(_FakeLeanPipeline):
        def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
            self.update_calls.append(fi)
            board = _make_board(COLOR_RED) if fi == 0 else Board.from_list(
                [[0] * BOARD_COLS for _ in range(BOARD_ROWS)],
            )
            state = BoardState.STABLE if fi == 0 else BoardState.MENU
            side = SimpleNamespace(
                state=state, score=100, confirmed_board=board,
                next_pair=None, dnext_pair=None, chain_event=None,
            )
            return SimpleNamespace(
                p1=side, p2=side, is_match_active=True, match_end_locked=False,
            )

    fake_pipeline = _FakeNoLockdownAttr()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
        )
    with np.load(out_npz) as data:
        assert list(data["post_match_lockdown_active"]) == [
            mod.POST_MATCH_LOCKDOWN_ACTIVE_UNKNOWN,
            mod.POST_MATCH_LOCKDOWN_ACTIVE_UNKNOWN,
        ]


# ============================
# STABLE持続confidenceタグ (2026-08-18 二次追加、収集ゲート→confidence
# タグへの役割転用)
# ============================

class TestStablePersistenceConfidenceColumn:
    """_LeanNpzAccumulator の stable_persistence_confidence 列の記録・保存を
    検証する (match_end_lockeds/post_match_lockdown_actives と同じ
    マーカー列方式)。"""

    def test_append_default_omitted_uses_unknown_sentinel(
        self, tmp_path: Path,
    ) -> None:
        """省略時は STABLE_PERSISTENCE_CONFIDENCE_UNKNOWN (-1、後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 10)
        assert acc.stable_persistence_confidences == [
            mod.STABLE_PERSISTENCE_CONFIDENCE_UNKNOWN
        ]
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert (
                data["stable_persistence_confidence"][0]
                == mod.STABLE_PERSISTENCE_CONFIDENCE_UNKNOWN
            )

    def test_append_explicit_values_roundtrip(self, tmp_path: Path) -> None:
        """True/False が npz に 1/0 でそのまま保存される。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(
            board._grid, "v1", "1P", 1.0, 0, 10,
            stable_persistence_confidence=True,
        )
        acc.append(
            board._grid, "v1", "1P", 2.0, 0, 11,
            stable_persistence_confidence=False,
        )
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert list(data["stable_persistence_confidence"]) == [1, 0]

    def test_existing_columns_unaffected(self, tmp_path: Path) -> None:
        """新列の追加が既存 match_end_locked/post_match_lockdown_active 列の
        挙動を変えないこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(
            board._grid, "v1", "1P", 1.0, 0, 10,
            match_end_locked=True, post_match_lockdown_active=False,
            stable_persistence_confidence=True,
        )
        out = tmp_path / "out.npz"
        acc.save(out)
        with np.load(out) as data:
            assert data["match_end_locked"][0] == 1
            assert data["post_match_lockdown_active"][0] == 0
            assert data["stable_persistence_confidence"][0] == 1


class _FakeCaptureLeanVaryingBrightness:
    """フレームごとに輝度が変わる cv2.VideoCapture フェイク (2026-08-18
    二次追加、stable_persistence_confidence 実配線検証用)。輝度差により
    src.board_motion の diff 計算が意味のある 0/1 混在を生む。"""

    def __init__(self, values: list[int], fps: float = 30.0) -> None:
        self._values = values
        self._fps = fps
        self._i = 0
        mod = _import_lean()
        self._h, self._w = mod.TARGET_H, mod.TARGET_W

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._values))
        return 0.0

    def set(self, prop: int, value: float) -> None:  # noqa: D401 - フェイクなので no-op
        pass

    def read(self) -> "tuple[bool, np.ndarray | None]":
        if self._i >= len(self._values):
            return False, None
        v = self._values[self._i]
        self._i += 1
        frame = np.full((self._h, self._w, 3), v, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        pass


class _FakeLeanPipelineStablePersistence(_FakeLeanPipeline):
    """2 フレームとも STABLE を返すが、盤面色をフレームごとに変えて dedup を
    回避するフェイク (2026-08-18 二次追加)。"""

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        self.update_calls.append(fi)
        color = COLOR_RED if fi == 0 else COLOR_BLUE
        board = _make_board(color)
        side = SimpleNamespace(
            state=BoardState.STABLE, score=100, confirmed_board=board,
            next_pair=None, dnext_pair=None, chain_event=None,
        )
        return SimpleNamespace(
            p1=side, p2=side, is_match_active=True, match_end_locked=False,
        )


def test_collect_lean_stable_persistence_confidence_does_not_block_emit(
    tmp_path: Path,
) -> None:
    """enable_stable_persistence_gate=True で confidence が 0/1 混在しても、
    いずれの snapshot も記録される (旧仕様=記録拒否は撤回済み、2026-08-18
    二次追加、STABLE持続ゲートの役割転用)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLeanVaryingBrightness([0, 200])
    fake_pipeline = _FakeLeanPipelineStablePersistence()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        n = mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
            enable_stable_persistence_gate=True,
        )
    # 2 フレーム (輝度 0 → 200 の急変) × 1P/2P = 4 行、いずれも拒否されない。
    assert n == 4
    with np.load(out_npz) as data:
        # frame0: 立ち上がり直後で diff 計算不能 → 保守的に True(1)。
        # frame1: 輝度急変で diff >= 閾値 → False(0)。emit は拒否されない。
        assert list(data["stable_persistence_confidence"]) == [1, 1, 0, 0]


def test_collect_lean_stable_persistence_confidence_unknown_when_gate_disabled(
    tmp_path: Path,
) -> None:
    """enable_stable_persistence_gate=False (既定) では計算自体を行わず、
    STABLE_PERSISTENCE_CONFIDENCE_UNKNOWN (-1) のまま保存される
    (bit-identical、2026-08-18 二次追加)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(2, fps=30.0)
    fake_pipeline = _FakeLeanPipelinePostMatchLockdown(lockdown_active=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
        )
    with np.load(out_npz) as data:
        assert list(data["stable_persistence_confidence"]) == [
            mod.STABLE_PERSISTENCE_CONFIDENCE_UNKNOWN,
            mod.STABLE_PERSISTENCE_CONFIDENCE_UNKNOWN,
        ]


# ============================
# 勝者パネルクロスチェック (W20/W21根治、2026-08-17 追加)
# ============================

class TestAssignWonLabelsPanelCrosscheck:
    """assign_won_labels の panel_winners 引数 (2 系統一致要求) を検証する。"""

    def test_panel_winners_none_keeps_legacy_score_only_behaviour(self) -> None:
        """panel_winners=None (既定) は従来通り score 系統単独判定 (bit-identical)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 1)
        acc.append(board._grid, "v1", "2P", 1.0, 0, 1)
        acc.assign_won_labels({0: {"1P": 5000, "2P": 3000}})
        assert acc.wons == [1.0, 0.0]

    def test_panel_agrees_with_score_keeps_winner(self) -> None:
        """score 系統とパネル系統が一致 → その勝者を採用する。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 1)
        acc.append(board._grid, "v1", "2P", 1.0, 0, 1)
        acc.assign_won_labels(
            {0: {"1P": 5000, "2P": 3000}}, panel_winners={0: "1P"},
        )
        assert acc.wons == [1.0, 0.0]

    def test_panel_disagrees_with_score_marks_unknown(self) -> None:
        """score 系統とパネル系統が不一致 → unknown (won は NaN のまま)。

        単一系統を無条件の正解にしない設計 (fail-silent 警戒)。
        """
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 1)
        acc.append(board._grid, "v1", "2P", 1.0, 0, 1)
        acc.assign_won_labels(
            {0: {"1P": 5000, "2P": 3000}}, panel_winners={0: "2P"},
        )
        assert all(math.isnan(w) for w in acc.wons)

    def test_panel_missing_marks_unknown_even_if_score_has_winner(self) -> None:
        """パネル系統が判定不能 (None) → score だけでは確定させない。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        board = _make_board(COLOR_RED)
        acc.append(board._grid, "v1", "1P", 1.0, 0, 1)
        acc.append(board._grid, "v1", "2P", 1.0, 0, 1)
        acc.assign_won_labels(
            {0: {"1P": 5000, "2P": 3000}}, panel_winners={0: None},
        )
        assert all(math.isnan(w) for w in acc.wons)

    def _suffocated_grid(self) -> "np.ndarray":
        """1P 窒息盤面 (row=1, col=2 にぷよ) の grid を返す (2026-08-19)。"""
        g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
        g[1][2] = COLOR_RED
        return Board.from_list(g)._grid

    def test_panel_unavailable_falls_back_to_survival_only(self) -> None:
        """PANEL_UNAVAILABLE (端点でパネル物理不可視、2026-08-19) は
        窒息判定のみで確定し、score 単独には緩和しないこと。

        score 系統は 1P 勝ちを示すが、盤面は 1P 窒息 (=2P 勝ち)。
        score 単独へ緩和していれば 1P 勝ちになるはずで、窒息判定 (2P) が
        採用されることが「緩和していない」ことの証明になる (score 単独緩和は
        断片化試合で 44.8% 誤ラベルの実測があるため禁止)。
        """
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(self._suffocated_grid(), "v1", "1P", 1.0, 0, 1)
        acc.append(_make_board(COLOR_BLUE)._grid, "v1", "2P", 1.0, 0, 1)
        acc.assign_won_labels(
            {0: {"1P": 5000, "2P": 3000}},
            panel_winners={0: mod.PANEL_UNAVAILABLE},
        )
        # 2P 勝ち → 1P 視点 won: 1P side=0.0 / 2P side=1.0
        assert acc.wons == [0.0, 1.0]

    def test_panel_unavailable_without_survival_stays_unknown(self) -> None:
        """PANEL_UNAVAILABLE かつ窒息判定も不能 → unknown (NaN のまま)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        acc.append(_make_board(COLOR_RED)._grid, "v1", "1P", 1.0, 0, 1)
        acc.append(_make_board(COLOR_BLUE)._grid, "v1", "2P", 1.0, 0, 1)
        acc.assign_won_labels(
            {0: {"1P": 5000, "2P": 3000}},
            panel_winners={0: mod.PANEL_UNAVAILABLE},
        )
        assert all(math.isnan(w) for w in acc.wons)


def test_detect_panel_winners_crosscheck_maps_unavailable_to_sentinel() -> None:
    """panel_unavailable=True の結果が番兵値 PANEL_UNAVAILABLE に、
    それ以外は従来通り winner にマップされること (2026-08-19)。"""
    mod = _import_lean()

    class _Result:
        def __init__(self, winner: str | None, unavailable: bool) -> None:
            self.winner = winner
            self.panel_unavailable = unavailable

    class _LegacyResult:
        """panel_unavailable フィールドを持たない旧 API 互換の結果。"""

        def __init__(self, winner: str | None) -> None:
            self.winner = winner

    class _Det:
        def detect_all_winners(
            self, cap: object, match_starts: list[float],
            last_observable_sec: float, offset_before: float = 1.0,
        ) -> list[object]:
            return [_Result(None, True), _Result("1P", False), _LegacyResult("2P")]

    class _DetCls:
        @classmethod
        def load_default(cls) -> _Det:
            return _Det()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod, "MatchWinnerDetector", _DetCls)
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: _FakeCaptureLean(1, fps=30.0))
        out = mod._detect_panel_winners_crosscheck(
            Path("dummy.mp4"), 0.0, [50.0, 100.0], 150.0,
        )
    assert out == {0: mod.PANEL_UNAVAILABLE, 1: "1P", 2: "2P"}


# ============================
# collect_lean() フル配線の結合テスト (W20/W21根治、2026-08-17 追加)
# ============================

class _FakeLeanPipelineVisualBoundary(_FakeLeanPipeline):
    """is_match_active の False→True 立ち上がりのみで境界が進むことを
    collect_lean() 経由のフル配線で検証するための最小フェイク。

    score は常に一定値 (100、リセットなし) にして score-reset 経路を
    完全に無効化し、視覚信号のみが game_idx を進めることを切り分ける。
    frame_idx 0-1: is_match_active=True、盤面 A (COLOR_RED)。
    frame_idx 2: is_match_active=False (試合外、盤面は COLOR_BLUE に変化)。
    frame_idx 3-17: is_match_active=True (立ち上がり、持続確認待ち区間
        BOUNDARY_VISUAL_RISE_PERSIST_SEC=0.5秒未満、W22根治 2026-08-17)、
        盤面は COLOR_GREEN。
    frame_idx 18-: 持続確認 (0.5秒経過、18/30fps=0.6秒) 済み後、盤面は
        COLOR_YELLOW (GREEN と区別し dedup で握り潰されないようにする。
        実運用でも新ゲームの盤面は持続確認窓の間に変化し続けるのが通常
        のため、この区別は実挙動を模する)。
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_active_by_frame = {0: True, 1: True, 2: False}

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        self.update_calls.append(fi)
        if fi < 2:
            color = COLOR_RED
        elif fi == 2:
            color = COLOR_BLUE
        elif fi < 18:
            color = COLOR_GREEN
        else:
            color = COLOR_YELLOW
        board = _make_board(color)
        side = SimpleNamespace(
            state=BoardState.STABLE, score=100, confirmed_board=board,
            next_pair=None, dnext_pair=None, chain_event=None,
        )
        return SimpleNamespace(
            p1=side, p2=side,
            is_match_active=self._is_active_by_frame.get(fi, True),
            match_end_locked=False,
        )


def test_collect_lean_boundary_multisignal_off_ignores_visual_transition(
    tmp_path: Path,
) -> None:
    """enable_boundary_multisignal=False (既定) では is_match_active の
    立ち上がりを無視し、score が変わらない限り game_idx は 0 のまま
    (旧挙動 bit-identical)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(21, fps=30.0)
    fake_pipeline = _FakeLeanPipelineVisualBoundary()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
        )
    with np.load(out_npz) as data:
        assert set(data["game_idx"].tolist()) == {0}
        # match_end_locked は fake が False を返すため 0 (未知 -1 ではない)
        assert set(data["match_end_locked"].tolist()) == {0}
    assert not (tmp_path / "out_boundary_anomalies.json").exists()


def test_collect_lean_boundary_multisignal_on_advances_game_idx(
    tmp_path: Path,
) -> None:
    """enable_boundary_multisignal=True で is_match_active 立ち上がりが
    持続確認 (0.5秒、W22根治) を経て game_idx を進める (score は一定値で
    リセットなし)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(21, fps=30.0)
    fake_pipeline = _FakeLeanPipelineVisualBoundary()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        out_npz = tmp_path / "out.npz"
        mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
            enable_boundary_multisignal=True,
        )
    with np.load(out_npz) as data:
        game_idxs = data["game_idx"]
        t_secs = data["t_sec"]
        # frame3 (立ち上がり、t=3/30) は持続確認完了前なのでまだ game 0
        # (frame18、t=18/30=0.6秒 で 0.5秒持続確認が完了して game 1 になる)
        pre = game_idxs[t_secs < 18 / 30]
        assert set(pre.tolist()) == {0}
        post = game_idxs[t_secs >= 18 / 30]
        assert set(post.tolist()) == {1}
    # score は一度もリセットしていないため異常イベントは発生しない
    assert not (tmp_path / "out_boundary_anomalies.json").exists()


class _FakeWinnerDetectorResult:
    """MatchWinnerDetector.detect_all_winners の戻り値要素の最小フェイク。"""

    def __init__(self, winner: str | None) -> None:
        self.winner = winner


class _FakeWinnerDetector:
    """MatchWinnerDetector の最小フェイク (cap を無視し固定結果を返す)。"""

    def __init__(self, winners: list[str | None]) -> None:
        self._winners = winners

    def detect_all_winners(
        self, cap: object, match_starts: list[float],
        last_observable_sec: float, offset_before: float = 1.0,
    ) -> list[_FakeWinnerDetectorResult]:
        return [_FakeWinnerDetectorResult(w) for w in self._winners]


def test_collect_lean_winner_panel_crosscheck_agrees_keeps_winner(
    tmp_path: Path,
) -> None:
    """パネル系統が score 系統と一致 → 従来通りの勝者ラベルを維持する。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(2, fps=30.0)
    fake_pipeline = _FakeLeanPipeline()  # 単一ゲーム、score=None (勝者None)

    class _FakeDetectorClass:
        @classmethod
        def load_default(cls) -> _FakeWinnerDetector:
            return _FakeWinnerDetector([None])  # score も None のため won は
            # 判定不能のままだが、配線経路 (panel_winners 生成→
            # assign_won_labels 呼び出し) が例外なく通ることを確認する。

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        mp.setattr(mod, "MatchWinnerDetector", _FakeDetectorClass)
        out_npz = tmp_path / "out.npz"
        n = mod.collect_lean(
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
            enable_winner_panel_crosscheck=True,
        )
    assert n == 0  # _FakeLeanPipeline は MENU 状態のため snapshot は 0 件


def test_collect_lean_winner_panel_crosscheck_failure_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """クロスチェック自体が失敗しても collect_lean は例外を投げず、
    score 系統単独判定にフォールバックする (fail-silent 警戒)。"""
    mod = _import_lean()
    fake_cap = _FakeCaptureLean(2, fps=30.0)
    fake_pipeline = _FakeLeanPipeline()

    class _FailingDetectorClass:
        @classmethod
        def load_default(cls) -> "_FailingDetectorClass":
            raise RuntimeError("boom (テスト用の意図的な失敗)")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mod.cv2, "VideoCapture", lambda _p: fake_cap)
        mp.setattr(RecognitionPipeline, "load_default", lambda *a, **kw: fake_pipeline)
        mp.setattr(mod, "MatchWinnerDetector", _FailingDetectorClass)
        out_npz = tmp_path / "out.npz"
        n = mod.collect_lean(  # 例外を投げずに完走すること自体が検証対象
            Path("dummy.mp4"), out_npz, sample_interval_frames=1,
            enable_winner_panel_crosscheck=True,
        )
    assert n == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


# ============================
# (d) STABLE持続確認 (2026-08-18、
# docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §5)
# ============================


def _make_frame_with_board_region(fill_value: int) -> np.ndarray:
    """盤面 ROI (1P/2P とも) を単色で塗った 1920x1080 フレームを作る。"""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :] = fill_value
    return frame


class TestUpdateRawPixelStable:
    """_update_raw_pixel_stable の gate 挙動を検証する。"""

    def test_disabled_returns_true_without_state_mutation(self) -> None:
        """既定 False では計算を一切行わず True を返す (bit-identical)。"""
        mod = _import_lean()
        state = mod._SideState()
        frame = _make_frame_with_board_region(100)
        result = mod._update_raw_pixel_stable(
            state, frame, "1P", 0.0, enable_stable_persistence_gate=False,
        )
        assert result is True
        assert state.motion_prev_gray is None
        assert state.motion_diffs == []

    def test_first_frame_returns_true_no_prev_gray(self) -> None:
        """直前フレームが無い最初の呼び出しは安全側 (True) を返す。"""
        mod = _import_lean()
        state = mod._SideState()
        frame = _make_frame_with_board_region(100)
        result = mod._update_raw_pixel_stable(
            state, frame, "1P", 0.0, enable_stable_persistence_gate=True,
        )
        assert result is True
        assert state.motion_prev_gray is not None

    def test_static_frames_stay_stable(self) -> None:
        """同一フレームが続く (静止) 間は True のまま。"""
        mod = _import_lean()
        state = mod._SideState()
        frame = _make_frame_with_board_region(100)
        for i in range(5):
            result = mod._update_raw_pixel_stable(
                state, frame, "1P", float(i) * 0.05,
                enable_stable_persistence_gate=True,
            )
        assert result is True

    def test_flashing_frames_become_unstable(self) -> None:
        """輝度が大きく変化するフレームが続くと False になる
        (連鎖フラッシュ/送付フラッシュ重畳の疑いを模擬)。
        """
        mod = _import_lean()
        state = mod._SideState()
        dark = _make_frame_with_board_region(20)
        bright = _make_frame_with_board_region(220)
        mod._update_raw_pixel_stable(
            state, dark, "1P", 0.0, enable_stable_persistence_gate=True,
        )
        result = mod._update_raw_pixel_stable(
            state, bright, "1P", 0.05, enable_stable_persistence_gate=True,
        )
        assert result is False

    def test_recovers_after_window_expires(self) -> None:
        """フラッシュ後、STABLE_PERSISTENCE_WINDOW_SEC 秒経過して窓から
        抜ければ (かつ以後静止すれば) 再び True になる。
        """
        mod = _import_lean()
        state = mod._SideState()
        dark = _make_frame_with_board_region(20)
        bright = _make_frame_with_board_region(220)
        mod._update_raw_pixel_stable(
            state, dark, "1P", 0.0, enable_stable_persistence_gate=True,
        )
        mod._update_raw_pixel_stable(
            state, bright, "1P", 0.05, enable_stable_persistence_gate=True,
        )
        # STABLE_PERSISTENCE_WINDOW_SEC (=0.25秒) を超えて静止し続ける。
        result = None
        for i in range(2, 12):
            t = 0.05 * i + 0.3
            result = mod._update_raw_pixel_stable(
                state, bright, "1P", t, enable_stable_persistence_gate=True,
            )
        assert result is True


# ============================
# CLI 配線: --enable-stable-persistence-gate
# ============================


def test_main_cli_enable_stable_persistence_gate_default_false() -> None:
    """CLI で --enable-stable-persistence-gate 未指定なら False が渡ること。

    (d) STABLE持続確認 (2026-08-18): backwards compat、既定 False で
    従来挙動と bit-identical。
    """
    captured = _run_fake_main_lean([])
    assert captured["enable_stable_persistence_gate"] is False


def test_main_cli_enable_stable_persistence_gate_flag_sets_true() -> None:
    """--enable-stable-persistence-gate 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-stable-persistence-gate"])
    assert captured["enable_stable_persistence_gate"] is True


# ============================
# CLI 配線: --enable-match-end-persist-override /
# --enable-post-match-lockdown-latch / --enable-result-screen-hardening
# (境界実装の仕上げ、2026-08-18)
# ============================


def test_main_cli_enable_match_end_persist_override_default_false() -> None:
    """CLI で --enable-match-end-persist-override 未指定なら False。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_match_end_persist_override"] is False


def test_main_cli_enable_match_end_persist_override_flag_sets_true() -> None:
    """--enable-match-end-persist-override 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-match-end-persist-override"])
    assert captured["enable_match_end_persist_override"] is True


def test_main_cli_enable_post_match_lockdown_latch_default_false() -> None:
    """CLI で --enable-post-match-lockdown-latch 未指定なら False。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_post_match_lockdown_latch"] is False


def test_main_cli_enable_post_match_lockdown_latch_flag_sets_true() -> None:
    """--enable-post-match-lockdown-latch 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-post-match-lockdown-latch"])
    assert captured["enable_post_match_lockdown_latch"] is True


def test_main_cli_enable_result_screen_hardening_default_false() -> None:
    """CLI で --enable-result-screen-hardening 未指定なら False。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_result_screen_hardening"] is False


def test_main_cli_enable_result_screen_hardening_flag_sets_true() -> None:
    """--enable-result-screen-hardening 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-result-screen-hardening"])
    assert captured["enable_result_screen_hardening"] is True


# ============================
# 連鎖中物理推論の配線 (2026-08-18、user確定要件)
# ============================


def test_main_cli_enable_chain_estimate_recording_default_false() -> None:
    """CLI で --enable-chain-estimate-recording 未指定なら False。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_chain_estimate_recording"] is False


def test_main_cli_enable_chain_estimate_recording_flag_sets_true() -> None:
    """--enable-chain-estimate-recording 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-chain-estimate-recording"])
    assert captured["enable_chain_estimate_recording"] is True


def test_main_cli_enable_move_segmented_recording_default_false() -> None:
    """CLI で --enable-move-segmented-recording 未指定なら False。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_move_segmented_recording"] is False


def test_main_cli_enable_move_segmented_recording_flag_sets_true() -> None:
    """--enable-move-segmented-recording 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-move-segmented-recording"])
    assert captured["enable_move_segmented_recording"] is True


def test_main_cli_enable_physics_persistence_filter_default_false() -> None:
    """CLI で --enable-physics-persistence-filter 未指定なら False。"""
    captured = _run_fake_main_lean([])
    assert captured["enable_physics_persistence_filter"] is False


def test_main_cli_enable_physics_persistence_filter_flag_sets_true() -> None:
    """--enable-physics-persistence-filter 指定時は True が渡ること。"""
    captured = _run_fake_main_lean(["--enable-physics-persistence-filter"])
    assert captured["enable_physics_persistence_filter"] is True


class TestChainEstimateRecording:
    """_process_side_lean の estimated_board 代替記録ロジックを検証する。"""

    def test_default_off_ignores_estimated_board_and_skips(self) -> None:
        """既定 False (enable_chain_estimate_recording 省略) では
        confirmed_board=None・CHAIN 中でも記録しない (bit-identical、
        従来挙動完全維持)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        estimated = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.CHAIN, 100, "v29", 1.0, 10,
            estimated_board=estimated, board_provenance="chain_estimate",
        )
        assert len(acc.grids) == 0

    def test_enabled_records_estimated_board_during_chain(self) -> None:
        """enable_chain_estimate_recording=True かつ confirmed_board=None・
        CHAIN 中なら estimated_board を代わりに記録し、board_provenance も
        伝搬すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        estimated = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.CHAIN, 100, "v29", 1.0, 10,
            estimated_board=estimated, board_provenance="chain_estimate",
            enable_chain_estimate_recording=True,
        )
        assert len(acc.grids) == 1
        assert np.array_equal(acc.grids[0], estimated._grid)
        assert acc.board_provenances[0] == "chain_estimate"

    def test_enabled_records_estimated_board_during_gravity_settle(self) -> None:
        """GRAVITY_SETTLE 中でも同様に estimated_board を記録すること。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        estimated = _make_board(COLOR_GREEN)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.GRAVITY_SETTLE, 100, "v29",
            1.0, 10,
            estimated_board=estimated, board_provenance="chain_estimate",
            enable_chain_estimate_recording=True,
        )
        assert len(acc.grids) == 1
        assert np.array_equal(acc.grids[0], estimated._grid)

    def test_enabled_rejects_low_confidence_provenance(self) -> None:
        """board_provenance == chain_estimate_low_confidence (起点誤認疑い)
        は enable_chain_estimate_recording=True でも採用しない。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        estimated = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.CHAIN, 100, "v29", 1.0, 10,
            estimated_board=estimated,
            board_provenance="chain_estimate_low_confidence",
            enable_chain_estimate_recording=True,
        )
        assert len(acc.grids) == 0

    def test_enabled_does_not_use_estimated_board_for_ojama_fall(self) -> None:
        """OJAMA_FALL 中は estimated_board が渡されていても記録しない
        (user明言「降り終わるまで待つ」。実運用では estimated_board 自体が
        常に None だが、本テストは念のため状態チェック自体も直接検証する)。
        """
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        estimated = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.OJAMA_FALL, 100, "v29", 1.0, 10,
            estimated_board=estimated, board_provenance="chain_estimate",
            enable_chain_estimate_recording=True,
        )
        assert len(acc.grids) == 0

    def test_enabled_does_not_override_real_observed_board(self) -> None:
        """confirmed_board が非 None (実測 STABLE) のときは
        enable_chain_estimate_recording=True でも estimated_board で
        上書きしない (実測が最優先)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        observed = _make_board(COLOR_RED)
        estimated = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", observed, BoardState.STABLE, 100, "v29", 1.0,
            10,
            estimated_board=estimated, board_provenance="observed",
            enable_chain_estimate_recording=True,
        )
        assert len(acc.grids) == 1
        assert np.array_equal(acc.grids[0], observed._grid)
        assert acc.board_provenances[0] == "observed"

    def test_board_provenance_recorded_regardless_of_flag(self) -> None:
        """board_provenance 列は enable_chain_estimate_recording の値に
        関わらず、実測 STABLE snapshot では常に記録される (npz に必ず残す
        という要件)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        observed = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", observed, BoardState.STABLE, 100, "v29", 1.0,
            10,
            board_provenance="observed",
        )
        assert acc.board_provenances[0] == "observed"

    def test_default_board_provenance_is_unknown_sentinel(self) -> None:
        """board_provenance を渡さない場合 (既定 None) は
        BOARD_PROVENANCE_UNKNOWN ("") で埋められる (後方互換)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
        )
        assert acc.board_provenances[0] == mod.BOARD_PROVENANCE_UNKNOWN

    def test_save_writes_board_provenance_key(self, tmp_path: Path) -> None:
        """save() が board_provenance 列を npz に書き出すこと (末尾追加、
        既存キー・キー順は不変)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board, BoardState.STABLE, 100, "v29", 1.0, 10,
            board_provenance="observed",
        )
        out = tmp_path / "test.npz"
        acc.save(out)
        data = np.load(out, allow_pickle=False)
        assert "board_provenance" in data
        assert data["board_provenance"][0] == "observed"

    def test_full_loop_default_fake_pipeline_no_crash_with_flag_on(
        self, tmp_path: Path,
    ) -> None:
        """estimated_board/board_provenance 属性を持たない旧式フェイク
        pipeline (_FakeLeanPipeline、SimpleNamespace) でも
        enable_chain_estimate_recording=True で AttributeError にならない
        こと (getattr フォールバック安全性の確認)。"""
        n, _ = _run_fake_collect_lean(
            tmp_path, 3, enable_chain_estimate_recording=True,
        )
        assert n == 0  # state=MENU のため元々何も記録されない

    def test_full_loop_records_estimated_board_via_collect_lean(
        self, tmp_path: Path,
    ) -> None:
        """collect_lean() の main loop を通しても estimated_board が
        npz に記録され、board_provenance 列も正しく保存されること
        (エンドツーエンド配線確認)。"""
        mod = _import_lean()
        estimated = _make_board(COLOR_BLUE)

        class _FakeChainEstimatePipeline(_FakeLeanPipeline):
            def update(self, fi: int, t_sec: float, frame: np.ndarray):
                self.update_calls.append(fi)
                side = SimpleNamespace(
                    state=BoardState.CHAIN, score=100, confirmed_board=None,
                    next_pair=None, dnext_pair=None, chain_event=None,
                    estimated_board=estimated, board_provenance="chain_estimate",
                )
                return SimpleNamespace(p1=side, p2=side)

        fake_cap = _FakeCaptureLean(1)

        def _fake_video_capture(_path: str) -> _FakeCaptureLean:
            return fake_cap

        fake_pipeline = _FakeChainEstimatePipeline()

        def _fake_load_default(*args: object, **kwargs: object):
            return fake_pipeline

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mod.cv2, "VideoCapture", _fake_video_capture)
            mp.setattr(RecognitionPipeline, "load_default", _fake_load_default)
            out_npz = tmp_path / "out.npz"
            n = mod.collect_lean(
                Path("dummy_video.mp4"), out_npz,
                enable_chain_estimate_recording=True,
            )
        assert n == 2  # 1P/2P 両方 (同一フェイク side を共有)
        data = np.load(out_npz, allow_pickle=False)
        assert np.array_equal(data["grids"][0], estimated._grid)
        assert data["board_provenance"][0] == "chain_estimate"


# ============================
# 1手区切り観測スケジューラ + 持続的物理制約フィルタ (2026-08-18)
# ============================


def _gravity_violation_board() -> Board:
    """col=0 の row=11 が浮遊 (row=12 が空) している合成盤面を返す。"""
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    g[11][0] = COLOR_RED  # row12 (最下段) が空なのに row11 に puyo → 重力違反
    return Board.from_list(g)


def _erasable_violation_board() -> Board:
    """2x2 の同色4連結 (消去可能グループ) を持つ合成盤面を返す。"""
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    g[11][0] = COLOR_RED
    g[11][1] = COLOR_RED
    g[12][0] = COLOR_RED
    g[12][1] = COLOR_RED
    return Board.from_list(g)


def _clean_board() -> Board:
    """物理制約違反のない合成盤面 (最下段を4色交互に敷き詰め、4連結なし)。"""
    g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
    colors = [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_RED, COLOR_BLUE]
    for col in range(BOARD_COLS):
        g[BOARD_ROWS - 1][col] = colors[col]
    return Board.from_list(g)


class TestMoveWindowScheduler:
    """_update_move_scheduler / _move_window_candidate_ok を検証する。"""

    def test_disabled_is_noop(self) -> None:
        """enable=False では state を一切変更しないこと (bit-identical)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), 1, BoardState.STABLE, 10, enable=False,
        )
        assert state.move_window_deadline_fi is None
        assert state.prev_next_pair is None
        assert state.prev_tsumo_count is None

    def test_next_pair_first_observation_does_not_open_window(self) -> None:
        """初回観測 (prev_next_pair が None) では窓を開かず、比較基準だけ記録する。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), None, BoardState.STABLE, 10, enable=True,
        )
        assert state.move_window_deadline_fi is None
        assert state.prev_next_pair == (1, 2)

    def test_next_pair_change_opens_window(self) -> None:
        """NEXT 繰り上がり (next_pair 変化) で猶予窓が開くこと。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), None, BoardState.STABLE, 10, enable=True,
        )
        mod._update_move_scheduler(
            state, (2, 3), None, BoardState.STABLE, 20, enable=True,
        )
        assert state.move_window_deadline_fi == 20 + mod.MOVE_SEGMENT_GRACE_FRAMES
        assert state.move_window_recorded is False

    def test_next_pair_no_change_does_not_reopen_window(self) -> None:
        """next_pair が変化しなければ窓は開かない (既存窓の状態も変えない)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), None, BoardState.STABLE, 10, enable=True,
        )
        mod._update_move_scheduler(
            state, (1, 2), None, BoardState.STABLE, 20, enable=True,
        )
        assert state.move_window_deadline_fi is None

    def test_tsumo_count_fallback_when_next_pair_none(self) -> None:
        """capture_next=False (next_pair=None) では tsumo_count 増分に
        フォールバックして窓を開くこと。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, None, 5, BoardState.STABLE, 10, enable=True,
        )
        mod._update_move_scheduler(
            state, None, 6, BoardState.STABLE, 21, enable=True,
        )
        assert state.move_window_deadline_fi == 21 + mod.MOVE_SEGMENT_GRACE_FRAMES

    def test_tsumo_count_decrease_does_not_open_window(self) -> None:
        """試合境界等で tsumo_count が減少しても窓は開かない (負の delta 無視)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, None, 10, BoardState.STABLE, 10, enable=True,
        )
        mod._update_move_scheduler(
            state, None, 0, BoardState.STABLE, 20, enable=True,
        )
        assert state.move_window_deadline_fi is None

    def test_ojama_fall_extends_open_window_deadline(self) -> None:
        """窓が開いている間、OJAMA_FALL 中は締切を押し戻すこと
        (user明言「降り終わるまで待つ」)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), None, BoardState.STABLE, 10, enable=True,
        )
        mod._update_move_scheduler(
            state, (2, 3), None, BoardState.STABLE, 20, enable=True,
        )
        first_deadline = state.move_window_deadline_fi
        mod._update_move_scheduler(
            state, (2, 3), None, BoardState.OJAMA_FALL, 40, enable=True,
        )
        assert state.move_window_deadline_fi == 40 + mod.MOVE_SEGMENT_GRACE_FRAMES
        assert state.move_window_deadline_fi > first_deadline

    def test_ojama_fall_without_open_window_does_nothing(self) -> None:
        """窓が開いていない状態で OJAMA_FALL が来ても窓を新設しないこと。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, None, None, BoardState.OJAMA_FALL, 40, enable=True,
        )
        assert state.move_window_deadline_fi is None

    def test_or_condition_tsumo_increment_opens_window_despite_unchanged_next_pair(
        self,
    ) -> None:
        """OR条件化 (2026-08-18 二次追補): NEXT ペアが偶然同色で変化しなくても

        tsumo_count 増分だけで窓が開くこと (4色パレットでの同色ペア衝突
        対策、coordinator指示)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), 5, BoardState.STABLE, 10, enable=True,
        )
        # next_pair は (1, 2) のまま変化しない (同色ペア衝突を模擬) が
        # tsumo_count は増分する
        mod._update_move_scheduler(
            state, (1, 2), 6, BoardState.STABLE, 20, enable=True,
        )
        assert state.move_window_deadline_fi == 20 + mod.MOVE_SEGMENT_GRACE_FRAMES
        assert state.move_window_recorded is False

    def test_or_condition_next_change_opens_window_despite_no_tsumo_increment(
        self,
    ) -> None:
        """OR条件化: tsumo_count が増分しなくても next_pair 変化だけで

        窓が開くこと (従来通りの経路も維持されていることの確認)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_move_scheduler(
            state, (1, 2), 5, BoardState.STABLE, 10, enable=True,
        )
        mod._update_move_scheduler(
            state, (2, 3), 5, BoardState.STABLE, 20, enable=True,
        )
        assert state.move_window_deadline_fi == 20 + mod.MOVE_SEGMENT_GRACE_FRAMES

    def test_or_condition_both_signals_fire_for_same_move_no_duplicate_via_dedup(
        self,
    ) -> None:
        """OR条件化での多重記録対策確認: NEXT変化とtsumo_count増分が別

        フレームで届いても (同一手を指す場合)、_should_emit の重複除外
        (grid_bytes 一致) により2回目は記録されないこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board_a = _make_board(COLOR_RED)
        # フレーム0: 初回観測 (比較基準の記録のみ、窓は開かない)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.MENU, None,
            "vid", 0.0, 0, next_pair=(1, 2), tsumo_count=5,
            enable_move_segmented_recording=True,
        )
        # フレーム10: tsumo_count 増分で窓が開く (NEXT は未変化)
        mod._process_side_lean(
            acc, state, "1P", board_a, BoardState.STABLE, 100,
            "vid", 10 / 30, 10, next_pair=(1, 2), tsumo_count=6,
            enable_move_segmented_recording=True,
        )
        assert len(acc.grids) == 1
        # フレーム20: 同じ手を指す NEXT 変化がやや遅れて到着 (窓は再オープン
        # されるが、盤面は変化していないので dedup が2回目の記録を阻止する)
        mod._process_side_lean(
            acc, state, "1P", board_a, BoardState.STABLE, 100,
            "vid", 20 / 30, 20, next_pair=(2, 3), tsumo_count=6,
            enable_move_segmented_recording=True,
        )
        assert len(acc.grids) == 1  # 二重記録されていないこと


class TestMoveWindowCandidateOk:
    """_move_window_candidate_ok の判定条件を検証する。"""

    def test_no_window_rejects(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        assert mod._move_window_candidate_ok(state, 10) is False

    def test_within_deadline_and_unrecorded_accepts(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        state.move_window_deadline_fi = 25
        state.move_window_recorded = False
        assert mod._move_window_candidate_ok(state, 20) is True

    def test_already_recorded_rejects(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        state.move_window_deadline_fi = 25
        state.move_window_recorded = True
        assert mod._move_window_candidate_ok(state, 20) is False

    def test_past_deadline_rejects(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        state.move_window_deadline_fi = 25
        state.move_window_recorded = False
        assert mod._move_window_candidate_ok(state, 26) is False

    def test_exact_deadline_frame_accepts(self) -> None:
        """締切ちょうどのフレームは猶予内として許容する (境界値)。"""
        mod = _import_lean()
        state = mod._SideState()
        state.move_window_deadline_fi = 25
        state.move_window_recorded = False
        assert mod._move_window_candidate_ok(state, 25) is True


class TestPhysicsPersistenceFilter:
    """_physics_violation_signature / _update_physics_transition_marker /

    _is_physics_violation_persistent を検証する。
    """

    def test_signature_empty_for_clean_board(self) -> None:
        mod = _import_lean()
        sim = mod.ChainSimulator()
        sig = mod._physics_violation_signature(sim, _clean_board())
        assert sig == frozenset()

    def test_signature_detects_gravity_violation(self) -> None:
        mod = _import_lean()
        sim = mod.ChainSimulator()
        sig = mod._physics_violation_signature(sim, _gravity_violation_board())
        assert (11, 0, "gravity") in sig

    def test_signature_detects_erasable_violation(self) -> None:
        mod = _import_lean()
        sim = mod.ChainSimulator()
        sig = mod._physics_violation_signature(sim, _erasable_violation_board())
        assert any(kind == "erasable" for _r, _c, kind in sig)

    def test_first_observation_never_persistent(self) -> None:
        """違反があっても初回観測 (比較対象なし) では棄却しないこと。"""
        mod = _import_lean()
        state = mod._SideState()
        sim = mod.ChainSimulator()
        rejected = mod._is_physics_violation_persistent(
            state, _gravity_violation_board(), sim,
        )
        assert rejected is False
        assert state.prev_violation_signature is not None

    def test_second_consecutive_same_violation_without_transition_rejected(
        self,
    ) -> None:
        """同一違反が正当な遷移を挟まず2回連続観測されたら棄却すること。"""
        mod = _import_lean()
        state = mod._SideState()
        sim = mod.ChainSimulator()
        board = _gravity_violation_board()
        mod._is_physics_violation_persistent(state, board, sim)
        rejected = mod._is_physics_violation_persistent(state, board, sim)
        assert rejected is True

    def test_legit_transition_clears_persistence(self) -> None:
        """間に正当な状態遷移 (TSUMO_FALL 等) を挟めば連続観測でも棄却しない
        こと (W24 教訓: 単純な連続回数閾値にしない)。"""
        mod = _import_lean()
        state = mod._SideState()
        sim = mod.ChainSimulator()
        board = _gravity_violation_board()
        mod._is_physics_violation_persistent(state, board, sim)
        mod._update_physics_transition_marker(state, BoardState.TSUMO_FALL, True)
        rejected = mod._is_physics_violation_persistent(state, board, sim)
        assert rejected is False

    def test_transition_marker_disabled_is_noop(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        mod._update_physics_transition_marker(state, BoardState.TSUMO_FALL, False)
        assert state.legit_transition_pending is False

    def test_transition_marker_tsumo_count_increment_sets_legit_transition(
        self,
    ) -> None:
        """tsumo_count 増分も正当な遷移として扱うこと (2026-08-18 是正、

        60fps stride-2 間引きで TSUMO_FALL 観測が漏れても検知できるようにする
        フォールバック信号)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_physics_transition_marker(
            state, BoardState.STABLE, True, tsumo_count=5,
        )
        assert state.legit_transition_pending is False  # 初回は比較対象なし
        mod._update_physics_transition_marker(
            state, BoardState.STABLE, True, tsumo_count=6,
        )
        assert state.legit_transition_pending is True

    def test_transition_marker_tsumo_count_independent_of_scheduler_field(
        self,
    ) -> None:
        """物理制約フィルタ単独使用 (enable_move_segmented_recording=False)

        でも tsumo_count 増分検知が機能すること (_update_move_scheduler の
        prev_tsumo_count とは独立フィールドで追跡するため)。"""
        mod = _import_lean()
        state = mod._SideState()
        mod._update_physics_transition_marker(
            state, BoardState.STABLE, True, tsumo_count=1,
        )
        mod._update_physics_transition_marker(
            state, BoardState.STABLE, True, tsumo_count=2,
        )
        assert state.legit_transition_pending is True
        assert state.prev_tsumo_count is None  # scheduler 側フィールドは無傷

    def test_different_violation_not_treated_as_persistent(self) -> None:
        """違反 signature が変われば (別セル/別種別) 単発扱いで棄却しないこと。"""
        mod = _import_lean()
        state = mod._SideState()
        sim = mod.ChainSimulator()
        mod._is_physics_violation_persistent(state, _gravity_violation_board(), sim)
        rejected = mod._is_physics_violation_persistent(
            state, _erasable_violation_board(), sim,
        )
        assert rejected is False

    def test_sim_none_never_rejects(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        assert mod._is_physics_violation_persistent(
            state, _gravity_violation_board(), None,
        ) is False


class TestShouldEmitMoveSegmentedAndPhysicsFilter:
    """_should_emit の新規ゲート (move-segmented / physics-persistence) を

    既存ゲート (STABLE/重複除外) と組み合わせて検証する。
    """

    def test_move_segmented_default_off_bit_identical(self) -> None:
        """新フラグを一切渡さない呼び出しは従来と完全に同じ挙動であること。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _clean_board()
        assert mod._should_emit(state, board, BoardState.STABLE) is True

    def test_move_segmented_no_window_rejects(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        board = _clean_board()
        assert mod._should_emit(
            state, board, BoardState.STABLE,
            enable_move_segmented_recording=True, frame_idx=10,
        ) is False

    def test_move_segmented_within_window_accepts(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        state.move_window_deadline_fi = 25
        board = _clean_board()
        assert mod._should_emit(
            state, board, BoardState.STABLE,
            enable_move_segmented_recording=True, frame_idx=20,
        ) is True

    def test_move_segmented_past_deadline_rejects(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        state.move_window_deadline_fi = 25
        board = _clean_board()
        assert mod._should_emit(
            state, board, BoardState.STABLE,
            enable_move_segmented_recording=True, frame_idx=30,
        ) is False

    def test_physics_filter_rejects_persistent_violation(self) -> None:
        mod = _import_lean()
        state = mod._SideState()
        sim = mod.ChainSimulator()
        board = _gravity_violation_board()
        # 1回目: 違反はあるが比較対象なしなので許容される
        assert mod._should_emit(
            state, board, BoardState.STABLE,
            enable_physics_persistence_filter=True, physics_sim=sim,
        ) is True
        state.last_emitted_grid = None  # 重複除外に引っかからないよう間に挟む
        # 2回目: 同一違反が正当な遷移なしで再度観測される → 棄却
        assert mod._should_emit(
            state, board, BoardState.STABLE,
            enable_physics_persistence_filter=True, physics_sim=sim,
        ) is False

    def test_physics_filter_default_off_bit_identical(self) -> None:
        """物理制約フィルタ無効時は違反があっても記録可否に影響しないこと。"""
        mod = _import_lean()
        state = mod._SideState()
        board = _gravity_violation_board()
        assert mod._should_emit(state, board, BoardState.STABLE) is True


class TestProcessSideLeanMoveSchedulerIntegration:
    """_process_side_lean 経由での1手区切りスケジューラ統合動作を検証する。"""

    def test_default_off_records_every_stable_change_like_before(
        self, tmp_path: Path,
    ) -> None:
        """新フラグ省略時は従来通り STABLE かつ非重複なら毎回記録されること
        (bit-identical、既存の event-driven 挙動を維持)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        board_a = _make_board(COLOR_RED)
        board_b = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", board_a, BoardState.STABLE, 100,
            "vid", 0.0, 0,
        )
        mod._process_side_lean(
            acc, state, "1P", board_b, BoardState.STABLE, 100,
            "vid", 1.0, 1,
        )
        assert len(acc.grids) == 2

    def test_move_segmented_records_only_first_stable_in_window(
        self, tmp_path: Path,
    ) -> None:
        """1手区切りモードでは、猶予窓内で最初に得られた STABLE のみ記録し、

        同一窓内の以降の STABLE 変化 (認識のブレ) は記録しないこと。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        # フレーム0: NEXT 初回観測 (窓は開かない)
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.MENU, None,
            "vid", 0.0, 0, next_pair=(1, 2),
            enable_move_segmented_recording=True,
        )
        # フレーム5: NEXT 繰り上がり → 猶予窓が開く (締切 = 5+15=20)
        board_a = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board_a, BoardState.STABLE, 100,
            "vid", 5 / 30, 5, next_pair=(2, 3),
            enable_move_segmented_recording=True,
        )
        assert len(acc.grids) == 1
        # フレーム8: 同一窓内で認識がブレて別盤面が観測されても記録しない
        board_b = _make_board(COLOR_BLUE)
        mod._process_side_lean(
            acc, state, "1P", board_b, BoardState.STABLE, 100,
            "vid", 8 / 30, 8, next_pair=(2, 3),
            enable_move_segmented_recording=True,
        )
        assert len(acc.grids) == 1
        assert np.array_equal(acc.grids[0], board_a._grid)

    def test_move_segmented_drops_move_when_grace_expires(
        self, tmp_path: Path,
    ) -> None:
        """猶予切れまで STABLE が得られなければその手は記録せず、

        次の手区切りへ回すこと (無理な穴埋めをしない)。"""
        mod = _import_lean()
        acc = mod._LeanNpzAccumulator()
        state = mod._SideState()
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.MENU, None,
            "vid", 0.0, 0, next_pair=(1, 2),
            enable_move_segmented_recording=True,
        )
        mod._process_side_lean(
            acc, state, "1P", None, BoardState.TSUMO_FALL, 100,
            "vid", 1 / 30, 1, next_pair=(2, 3),
            enable_move_segmented_recording=True,
        )
        # 猶予切れ後 (frame_idx=1+15+1=17) に STABLE が得られても記録しない
        board_a = _make_board(COLOR_RED)
        mod._process_side_lean(
            acc, state, "1P", board_a, BoardState.STABLE, 100,
            "vid", 17 / 30, 17, next_pair=(2, 3),
            enable_move_segmented_recording=True,
        )
        assert len(acc.grids) == 0
        # 次の手区切り (NEXT 再度繰り上がり) では新しい窓で記録できる
        mod._process_side_lean(
            acc, state, "1P", board_a, BoardState.STABLE, 100,
            "vid", 20 / 30, 20, next_pair=(3, 4),
            enable_move_segmented_recording=True,
        )
        assert len(acc.grids) == 1


# ============================
# 新試合証拠ゲート (2026-08-19、user指示「必ず試合前スコアは0」)
# ============================

class TestBoundaryNewMatchEvidence:
    """observe_visual_signal の require_newmatch_evidence ゲートの検証。

    背景: 視覚立ち上がりだけで境界を確定すると、試合中の is_active 乱れ
    (ラッチ解除失敗等) が偽境界を量産する (実測: 50本で試合総数+50%断片化、
    won欠損38.3%、欠損試合直後ギャップ中央値0.6秒 vs 正常な試合間10.1秒)。
    """

    def test_default_off_ignores_evidence_argument(self) -> None:
        """require_newmatch_evidence=False (既定) では new_match_evidence を
        渡しても無視され、従来通り立ち上がりだけで境界が進む (bit-identical)。
        """
        mod = _import_lean()
        shared = mod._SharedGameCounter(multisignal_mode=True)
        shared.observe_visual_signal(False, t_sec=1.0, new_match_evidence=False)
        shared.observe_visual_signal(True, t_sec=2.0, new_match_evidence=False)
        shared.observe_visual_signal(
            True, t_sec=2.0 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
            new_match_evidence=False,
        )
        assert shared.game_idx == 1
        assert shared.rejected_rise_times == []

    def test_evidence_present_at_confirm_advances_immediately(self) -> None:
        """ゲート ON: 持続確認完了時点で証拠があれば従来と同じく即確定する。
        境界時刻は立ち上がり本来の時刻 (2.0)。"""
        mod = _import_lean()
        shared = mod._SharedGameCounter(
            multisignal_mode=True, require_newmatch_evidence=True,
        )
        shared.observe_visual_signal(False, t_sec=1.0, new_match_evidence=False)
        shared.observe_visual_signal(True, t_sec=2.0, new_match_evidence=True)
        shared.observe_visual_signal(
            True, t_sec=2.0 + mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC,
            new_match_evidence=True,
        )
        assert shared.game_idx == 1
        assert shared.last_visual_rise_sec == 2.0

    def test_evidence_arriving_within_window_confirms_boundary(self) -> None:
        """ゲート ON: 持続確認完了時に証拠が無くても、証拠窓以内に証拠が
        観測されれば境界を確定する (score OCR/盤面確定のラグ吸収)。
        境界時刻は立ち上がり本来の時刻を維持する。"""
        mod = _import_lean()
        persist = mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC
        shared = mod._SharedGameCounter(
            multisignal_mode=True, require_newmatch_evidence=True,
        )
        shared.observe_visual_signal(False, t_sec=1.0, new_match_evidence=False)
        shared.observe_visual_signal(True, t_sec=2.0, new_match_evidence=False)
        shared.observe_visual_signal(
            True, t_sec=2.0 + persist, new_match_evidence=False,
        )
        assert shared.game_idx == 0, "証拠が出るまでは確定しない"
        shared.observe_visual_signal(
            True, t_sec=2.0 + persist + 1.0, new_match_evidence=True,
        )
        assert shared.game_idx == 1
        assert shared.last_visual_rise_sec == 2.0, (
            "境界時刻は証拠観測時刻でなく立ち上がり時刻を使う"
        )

    def test_no_evidence_within_window_rejects_false_boundary(self) -> None:
        """ゲート ON: 証拠窓が切れるまで証拠が出なければ偽境界として破棄
        (試合中の is_active 乱れは境界にならない)。破棄記録が残る。"""
        mod = _import_lean()
        persist = mod.BOUNDARY_VISUAL_RISE_PERSIST_SEC
        window = mod.BOUNDARY_NEWMATCH_EVIDENCE_WINDOW_SEC
        shared = mod._SharedGameCounter(
            multisignal_mode=True, require_newmatch_evidence=True,
        )
        shared.observe_visual_signal(False, t_sec=1.0, new_match_evidence=False)
        shared.observe_visual_signal(True, t_sec=2.0, new_match_evidence=False)
        shared.observe_visual_signal(
            True, t_sec=2.0 + persist + window + 0.1, new_match_evidence=False,
        )
        assert shared.game_idx == 0, "証拠なしの立ち上がりは境界にならない"
        assert shared.rejected_rise_times == [2.0]
        # 破棄後、後続の本物の立ち上がり (証拠つき) は独立に確定できる
        shared.observe_visual_signal(
            False, t_sec=20.0, new_match_evidence=False,
        )
        shared.observe_visual_signal(True, t_sec=21.0, new_match_evidence=True)
        shared.observe_visual_signal(
            True, t_sec=21.0 + persist, new_match_evidence=True,
        )
        assert shared.game_idx == 1

    def test_compute_newmatch_evidence_score_zero(self) -> None:
        """_compute_newmatch_evidence: 両者スコア数値0で True、片側のみ0や
        非0では False。"""
        mod = _import_lean()

        class _Side:
            def __init__(self, score, board) -> None:
                self.score = score
                self.confirmed_board = board

        class _Res:
            def __init__(self, s1, s2) -> None:
                self.p1 = s1
                self.p2 = s2

        assert mod._compute_newmatch_evidence(
            _Res(_Side(0, None), _Side(0, None))
        ) is True
        assert mod._compute_newmatch_evidence(
            _Res(_Side(0, None), _Side(4800, None))
        ) is False
        assert mod._compute_newmatch_evidence(
            _Res(_Side(None, None), _Side(0, None))
        ) is False

    def test_compute_newmatch_evidence_empty_boards(self) -> None:
        """_compute_newmatch_evidence: スコアが0でなくても両者の確定盤面が
        ほぼ空 (<= NEW_MATCH_BOARD_MAX_PUYOS) なら True。None 盤面や
        試合中盤面 (多数のぷよ) は False。"""
        mod = _import_lean()
        from src.board import Board

        empty = Board()
        mid = Board()
        for r in range(6, 13):
            for c in range(6):
                mid.set(r, c, 1)

        class _Side:
            def __init__(self, score, board) -> None:
                self.score = score
                self.confirmed_board = board

        class _Res:
            def __init__(self, s1, s2) -> None:
                self.p1 = s1
                self.p2 = s2

        assert mod._compute_newmatch_evidence(
            _Res(_Side(4800, empty), _Side(3000, empty))
        ) is True
        assert mod._compute_newmatch_evidence(
            _Res(_Side(4800, None), _Side(3000, empty))
        ) is False, "None 盤面 (試合中の一時クリア) は空盤面証拠に使わない"
        assert mod._compute_newmatch_evidence(
            _Res(_Side(4800, mid), _Side(3000, empty))
        ) is False, "試合中盤面 (多数のぷよ) は証拠にならない"
