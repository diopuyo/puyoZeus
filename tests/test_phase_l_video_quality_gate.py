"""scripts/phase_l_video_quality_gate.py の単体テスト。

2026-08-11 追加: 品質ゲート検査1(試合開始直後の空盤面)のパターンB対策
(試合開始代理値のズレ = score リセット誤発火で窓が中盤に当たる問題)の
頑健化ロジックを検証する。パターンA (score 未読み取り = 非試合画面疑いの
本物汚染) は従来通り検出され続けることも合わせて確認する
(過修正防止、memory `project_quality_gate_fail_triage_2026-08-08` 参照)。
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.phase_l_video_quality_gate import (
    MATCH_START_ANCHOR_SCORE_MAX,
    MATCH_START_OFFSET_HI_SEC,
    MATCH_START_OFFSET_LO_SEC,
    MATCH_START_ROW_HI_EXCLUSIVE,
    MATCH_START_ROW_LO,
    SCORE_UNREADABLE,
    VideoArrays,
    VideoGateResult,
    _anchor_score,
    _is_untrustworthy_start,
    compute_match_start_nonempty,
    load_video_arrays,
)
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY

_BAND_ROWS = MATCH_START_ROW_HI_EXCLUSIVE - MATCH_START_ROW_LO
_BAND_CELLS = _BAND_ROWS * BOARD_COLS


def _empty_grid() -> np.ndarray:
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)


def _full_grid(color: int = 1) -> np.ndarray:
    g = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    g[MATCH_START_ROW_LO:MATCH_START_ROW_HI_EXCLUSIVE, :] = color
    return g


# ============================
# _anchor_score
# ============================

class TestAnchorScore:
    def test_score_column_missing_returns_none(self) -> None:
        """score 列が無い (旧形式 npz) 場合は None を返す (後方互換の要)。"""
        arrays = VideoArrays(
            video_id="v",
            grids=np.zeros((2, BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            t_sec=np.array([0.0, 1.0]),
            side=np.array(["1P", "1P"]),
            game_idx=np.array([0, 0]),
        )
        mask = np.array([True, True])
        assert _anchor_score(arrays, mask, start_sec=0.0) is None

    def test_no_matching_entries_returns_none(self) -> None:
        arrays = VideoArrays(
            video_id="v",
            grids=np.zeros((2, BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            t_sec=np.array([0.0, 1.0]),
            side=np.array(["1P", "1P"]),
            game_idx=np.array([0, 0]),
            score=np.array([0, 0]),
        )
        mask = np.array([False, False])
        assert _anchor_score(arrays, mask, start_sec=0.0) is None

    def test_returns_nearest_in_time_score(self) -> None:
        arrays = VideoArrays(
            video_id="v",
            grids=np.zeros((3, BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            t_sec=np.array([10.0, 11.0, 12.0]),
            side=np.array(["1P", "1P", "1P"]),
            game_idx=np.array([0, 0, 0]),
            score=np.array([0, 5, 999]),
        )
        mask = np.array([True, True, True])
        assert _anchor_score(arrays, mask, start_sec=11.2) == 5


# ============================
# _is_untrustworthy_start
# ============================

class TestIsUntrustworthyStart:
    def test_none_is_trusted(self) -> None:
        """score 情報が無い (判定不能) 場合は除外しない (従来挙動を維持)。"""
        assert _is_untrustworthy_start(None) is False

    def test_unreadable_sentinel_is_trusted(self) -> None:
        """score 未読み取り (-1) はパターンA疑いのため除外しない。"""
        assert _is_untrustworthy_start(SCORE_UNREADABLE) is False

    def test_near_zero_score_is_trusted(self) -> None:
        assert _is_untrustworthy_start(0) is False

    def test_boundary_value_is_trusted(self) -> None:
        assert _is_untrustworthy_start(MATCH_START_ANCHOR_SCORE_MAX) is False

    def test_above_threshold_is_untrustworthy(self) -> None:
        assert _is_untrustworthy_start(MATCH_START_ANCHOR_SCORE_MAX + 1) is True

    def test_large_midgame_score_is_untrustworthy(self) -> None:
        assert _is_untrustworthy_start(8675) is True


# ============================
# compute_match_start_nonempty (統合)
# ============================

class TestComputeMatchStartNonempty:
    def _build_arrays(self, with_score: bool = True) -> VideoArrays:
        """3 game_idx の合成シナリオ。

        game=0: 真の試合開始 (score=0近傍) で盤面も空 -> 正常に評価に寄与
        game=1: パターンB (score リセット誤発火疑い、score=9999で満杯盤面)
                -> 頑健化ロジックにより除外されるべき
        game=2: パターンA (score 未読み取り=-1、満杯盤面)
                -> 除外せず引き続き検出対象に残るべき
        """
        grids = np.stack([
            _empty_grid(), _empty_grid(),  # game 0: anchor(t=0) empty, window(t=2) empty
            _full_grid(), _full_grid(),    # game 1: anchor(t=100) full, window(t=102) full
            _full_grid(), _full_grid(),    # game 2: anchor(t=200) full, window(t=202) full
        ])
        t_sec = np.array([0.0, 2.0, 100.0, 102.0, 200.0, 202.0])
        side = np.array(["1P"] * 6)
        game_idx = np.array([0, 0, 1, 1, 2, 2])
        score = np.array([0, 0, 9999, 9999, SCORE_UNREADABLE, SCORE_UNREADABLE])
        kwargs = dict(
            video_id="synthetic",
            grids=grids,
            t_sec=t_sec,
            side=side,
            game_idx=game_idx,
        )
        if with_score:
            kwargs["score"] = score
        return VideoArrays(**kwargs)

    def test_pattern_b_window_excluded_pattern_a_window_kept(self) -> None:
        arrays = self._build_arrays(with_score=True)
        rate, n_cells, n_excluded = compute_match_start_nonempty(arrays)

        # game=1 (パターンB) が除外され、game=0 と game=2 のみが分母に入る。
        assert n_excluded == 1
        assert n_cells == 2 * _BAND_CELLS
        # game=0 (空盤面) は非空0、game=2 (パターンA満杯盤面) は全セル非空。
        assert rate == pytest.approx(_BAND_CELLS / (2 * _BAND_CELLS))

    def test_without_score_column_falls_back_to_legacy_behavior(self) -> None:
        """score 列が無い npz (旧形式) では誰も除外されない (後方互換)。"""
        arrays = self._build_arrays(with_score=False)
        rate, n_cells, n_excluded = compute_match_start_nonempty(arrays)

        assert n_excluded == 0
        assert n_cells == 3 * _BAND_CELLS
        # game=0 は空盤面、game=1/2 は満杯盤面 (旧挙動では両方カウントされる)。
        assert rate == pytest.approx((2 * _BAND_CELLS) / (3 * _BAND_CELLS))

    def test_empty_video_returns_nan(self) -> None:
        arrays = VideoArrays(
            video_id="v",
            grids=np.zeros((0, BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            t_sec=np.array([]),
            side=np.array([]),
            game_idx=np.array([]),
        )
        rate, n_cells, n_excluded = compute_match_start_nonempty(arrays)
        assert np.isnan(rate)
        assert n_cells == 0
        assert n_excluded == 0


# ============================
# 後方互換性
# ============================

class TestBackwardsCompat:
    def test_video_arrays_constructs_without_score_kwarg(self) -> None:
        """score 未指定でも構築可能 (既存呼出元を壊さない)。"""
        arrays = VideoArrays(
            video_id="v",
            grids=np.zeros((1, BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            t_sec=np.array([0.0]),
            side=np.array(["1P"]),
            game_idx=np.array([0]),
        )
        assert arrays.score.size == 0

    def test_video_gate_result_defaults_excluded_windows_to_zero(self) -> None:
        """match_start_excluded_windows 未指定でも構築可能 (既存呼出元互換)。"""
        result = VideoGateResult(
            video_id="v",
            match_start_nonempty_rate=0.0,
            match_start_n_cells=10,
            systemic_bias_max_z=0.0,
            systemic_bias_worst_combo="col=0,color=1",
            col_unknown_max_rate=0.0,
            avg_puyo_count=30.0,
            ojama_diag_decrease_no_chain_rate=float("nan"),
            verdict="PASS",
        )
        assert result.match_start_excluded_windows == 0

    def test_load_video_arrays_without_score_key(self, tmp_path) -> None:
        """score キーの無い旧形式 npz でも読み込める。"""
        npz_path = tmp_path / "legacy.npz"
        np.savez(
            npz_path,
            grids=np.zeros((1, BOARD_ROWS, BOARD_COLS), dtype=np.int8),
            t_sec=np.array([0.0]),
            side=np.array(["1P"]),
            game_idx=np.array([0]),
        )
        arrays = load_video_arrays(npz_path)
        assert arrays.score.size == 0
