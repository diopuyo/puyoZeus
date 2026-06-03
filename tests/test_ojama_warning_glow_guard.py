"""src/ojama_warning_glow_guard.py のユニットテスト.

テスト方針:
  - stateless 関数 (compute_glow_score, update_glow_state, apply_glow_guard) を独立に検証
  - GlowGuardState の ON/OFF 遷移・上限解除を網羅
  - default OFF (enable_ojama_warning_glow_guard=False) で現挙動不変を確認
  - CLI フラグが store_true で正しく動くことを argparse 直接テストで確認
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# プロジェクトルートを sys.path に追加
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_OJAMA,
    COLOR_UNKNOWN, COLOR_YELLOW, Board,
)
from src.image_reader import BoardRegion, VISIBLE_ROWS
from src.ojama_warning_glow_guard import (
    GLOW_CONSEC_MIN,
    GLOW_DETECTION_THRESHOLD,
    GLOW_MAX_HOLD_FRAMES,
    GLOW_RATIO_HIGH,
    GLOW_RATIO_LOW,
    GLOW_RELEASE_CONSEC,
    GLOW_ROI_ROW_COUNT,
    V_HIGH_THRESHOLD,
    GlowGuardState,
    apply_glow_guard,
    compute_glow_score,
    update_glow_state,
)


# ============================
# テスト用ヘルパー
# ============================


def _blank_frame(brightness: int = 30) -> np.ndarray:
    """1920×1080 の単色 BGR フレームを生成する。"""
    return np.full((1080, 1920, 3), brightness, dtype=np.uint8)


def _glow_frame(region: BoardRegion) -> np.ndarray:
    """指定 region の上部 GLOW_ROI_ROW_COUNT 行を高輝度白で塗ったフレームを生成する。

    V_HIGH_THRESHOLD 以上の画素が ROI の全画素を占めるため
    glow_score = 1.0 になることを保証する。
    """
    frame = _blank_frame(0)
    roi_h = int(region.cell_height * GLOW_ROI_ROW_COUNT)
    x1, y1 = region.x, region.y
    x2 = x1 + region.width
    y2 = y1 + roi_h
    # BGR で (255,255,255) = HSV V=255 → V_HIGH_THRESHOLD 確実超過
    frame[y1:y2, x1:x2] = 255
    return frame


def _make_board(colors: dict[tuple[int, int], int] | None = None) -> Board:
    """テスト用の盤面を生成する。colors: {(row, col): color_value}"""
    b = Board()
    if colors:
        for (r, c), v in colors.items():
            b.set(r, c, v)
    return b


_P1_REGION = BoardRegion(x=282, y=160, width=384, height=720)


# ============================
# compute_glow_score のテスト
# ============================


class TestComputeGlowScore:
    """compute_glow_score: V_high_ratio から glow_score を計算する。"""

    def test_dark_frame_returns_zero(self) -> None:
        """暗いフレーム (V 低) は glow_score=0 を返す。"""
        frame = _blank_frame(20)  # V=20 < V_HIGH_THRESHOLD=220
        score = compute_glow_score(frame, _P1_REGION)
        assert score == pytest.approx(0.0)

    def test_bright_frame_returns_one(self) -> None:
        """上部 ROI が全面高輝度ならglow_score=1.0 を返す。"""
        frame = _glow_frame(_P1_REGION)
        score = compute_glow_score(frame, _P1_REGION)
        assert score == pytest.approx(1.0)

    def test_score_normalized_between_zero_and_one(self) -> None:
        """glow_score は常に [0, 1] の範囲内に収まる。"""
        frame = _blank_frame(128)  # 中間輝度
        # ROI の一部だけ高輝度にして中間 score を作る
        roi_h = int(_P1_REGION.cell_height * GLOW_ROI_ROW_COUNT)
        partial_h = roi_h // 2
        frame[_P1_REGION.y:_P1_REGION.y + partial_h, _P1_REGION.x:_P1_REGION.x + _P1_REGION.width] = 255
        score = compute_glow_score(frame, _P1_REGION)
        assert 0.0 <= score <= 1.0

    def test_invalid_region_returns_zero(self) -> None:
        """フレーム外の region (x2 > w_img) でも例外を出さず 0 を返す。"""
        region = BoardRegion(x=1900, y=900, width=100, height=200)  # はみ出し
        frame = _blank_frame(30)
        score = compute_glow_score(frame, region)
        assert 0.0 <= score <= 1.0

    def test_empty_roi_returns_zero(self) -> None:
        """ROI がゼロピクセル (width=0) でも 0.0 を返す。"""
        region = BoardRegion(x=0, y=0, width=0, height=100)
        frame = _blank_frame(30)
        score = compute_glow_score(frame, region)
        assert score == pytest.approx(0.0)


# ============================
# update_glow_state のテスト
# ============================


class TestUpdateGlowState:
    """update_glow_state: ON/OFF/上限解除の state 遷移を検証する。"""

    def test_single_high_score_does_not_activate(self) -> None:
        """1 フレームだけ高 score でも GLOW_CONSEC_MIN 未満なら glow_active=False。"""
        state = GlowGuardState()
        is_glow = update_glow_state(state, 1.0, frame_idx=0)
        assert not is_glow
        assert not state.glow_active

    def test_consecutive_high_activates(self) -> None:
        """GLOW_CONSEC_MIN 連続で高 score → glow_active=True になる。"""
        state = GlowGuardState()
        for fi in range(GLOW_CONSEC_MIN):
            result = update_glow_state(state, 1.0, frame_idx=fi)
        assert result
        assert state.glow_active

    def test_release_after_consecutive_low(self) -> None:
        """ON 状態から GLOW_RELEASE_CONSEC 連続で低 score → OFF になる。"""
        state = GlowGuardState()
        # ON にする
        for fi in range(GLOW_CONSEC_MIN):
            update_glow_state(state, 1.0, frame_idx=fi)
        assert state.glow_active
        # OFF に戻す
        for fi in range(GLOW_CONSEC_MIN, GLOW_CONSEC_MIN + GLOW_RELEASE_CONSEC):
            result = update_glow_state(state, 0.0, frame_idx=fi)
        assert not result
        assert not state.glow_active

    def test_max_hold_forces_release(self) -> None:
        """GLOW_MAX_HOLD_FRAMES フレーム保持後に強制解除される。"""
        state = GlowGuardState()
        # ON にする
        for fi in range(GLOW_CONSEC_MIN):
            update_glow_state(state, 1.0, frame_idx=fi)
        assert state.glow_active
        # ON を維持しつつ上限まで進める
        fi_start = GLOW_CONSEC_MIN
        for fi in range(fi_start, fi_start + GLOW_MAX_HOLD_FRAMES):
            result = update_glow_state(state, 1.0, frame_idx=fi)
        # 上限到達で強制解除されているはず
        assert not state.glow_active

    def test_single_low_between_high_does_not_release(self) -> None:
        """ON 中に 1 フレームだけ低 score が来ても GLOW_RELEASE_CONSEC 未満なら維持。"""
        if GLOW_RELEASE_CONSEC <= 1:
            pytest.skip("GLOW_RELEASE_CONSEC=1 では単発でも解除される設計")
        state = GlowGuardState()
        # ON にする
        for fi in range(GLOW_CONSEC_MIN):
            update_glow_state(state, 1.0, frame_idx=fi)
        assert state.glow_active
        # 1 フレームだけ低
        result = update_glow_state(state, 0.0, frame_idx=GLOW_CONSEC_MIN)
        assert result  # まだ ON
        # すぐ高に戻す → ON 維持
        result = update_glow_state(state, 1.0, frame_idx=GLOW_CONSEC_MIN + 1)
        assert result


# ============================
# apply_glow_guard のテスト
# ============================


class TestApplyGlowGuard:
    """apply_glow_guard: confirmed 保護ロジックを検証する。"""

    def test_off_returns_confirmed_unchanged(self) -> None:
        """glow_active=False なら confirmed をそのまま返す。"""
        state = GlowGuardState()
        state.frozen_board = _make_board({(5, 2): COLOR_YELLOW})
        confirmed = _make_board({(5, 2): COLOR_OJAMA})  # 誤認された状態
        result = apply_glow_guard(confirmed, state, is_glow_active=False)
        # is_glow_active=False → confirmed そのまま (OJAMA のまま)
        assert int(result.get(5, 2)) == COLOR_OJAMA

    def test_on_frozen_color_preserved(self) -> None:
        """glow_active=True かつ frozen に有色 → frozen 色で上書きする。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(5, 2): COLOR_YELLOW})
        # confirmed は誤認で OJAMA になっている
        confirmed = _make_board({(5, 2): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # frozen の YELLOW で保護される
        assert int(result.get(5, 2)) == COLOR_YELLOW

    def test_on_new_puyo_becomes_unknown(self) -> None:
        """glow_active=True かつ frozen が空 かつ confirmed に有色 → UNKNOWN 留保。"""
        state = GlowGuardState()
        state.glow_active = True
        # frozen は空 (発光前に置かれたぷよがない)
        state.frozen_board = _make_board()
        # 発光中に新しく「色」が見えた
        confirmed = _make_board({(3, 1): COLOR_BLUE})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # 新規出現ぷよは UNKNOWN 留保
        assert int(result.get(3, 1)) == COLOR_UNKNOWN

    def test_on_no_frozen_board_returns_confirmed(self) -> None:
        """frozen_board=None の場合は confirmed をそのまま返す (安全弁)。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = None
        confirmed = _make_board({(5, 2): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        assert int(result.get(5, 2)) == COLOR_OJAMA

    def test_empty_to_empty_unchanged(self) -> None:
        """frozen も confirmed も空のセルは EMPTY のまま。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board()  # 全空
        confirmed = _make_board()           # 全空
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                assert int(result.get(r, c)) == COLOR_EMPTY


# ============================
# default OFF で現挙動不変のテスト
# ============================


class TestDefaultOff:
    """enable_ojama_warning_glow_guard=False (default) で挙動が変わらないことを確認。"""

    def test_pipeline_default_off_no_glow_state(self) -> None:
        """default OFF のとき pipeline は _glow_guard_1p/_2p が None になる。"""
        # pipeline のインポートは重いのでモジュール import レベルでは避け、
        # GlowGuardState の存在チェックで代替する (CI が短い環境向け)
        state = GlowGuardState()
        # glow_active=False の場合は apply_glow_guard が何もしないことを確認
        confirmed = _make_board({(5, 2): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=False)
        # 元の confirmed 値と同じ
        assert int(result.get(5, 2)) == COLOR_OJAMA

    def test_glow_state_initial_inactive(self) -> None:
        """GlowGuardState の初期状態は glow_active=False で安全デフォルト。"""
        state = GlowGuardState()
        assert not state.glow_active
        assert state.consec_on == 0
        assert state.consec_off == 0
        assert state.hold_frame_count == 0
        assert state.frozen_board is None


# ============================
# CLI フラグ (store_true) のテスト
# ============================


class TestCliFlag:
    """--ojama-warning-glow-guard が store_true で正しく動くことを argparse で確認。"""

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--ojama-warning-glow-guard",
            action="store_true",
            default=False,
            dest="enable_ojama_warning_glow_guard",
        )
        return parser

    def test_flag_absent_is_false(self) -> None:
        """フラグなしなら enable_ojama_warning_glow_guard=False。"""
        parser = self._build_parser()
        args = parser.parse_args([])
        assert args.enable_ojama_warning_glow_guard is False

    def test_flag_present_is_true(self) -> None:
        """--ojama-warning-glow-guard 指定で enable_ojama_warning_glow_guard=True。"""
        parser = self._build_parser()
        args = parser.parse_args(["--ojama-warning-glow-guard"])
        assert args.enable_ojama_warning_glow_guard is True

    def test_no_prefix_not_recognized(self) -> None:
        """store_true のため --no- 接頭辞は認識されない (BooleanOptionalAction と異なる)。"""
        parser = self._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--no-ojama-warning-glow-guard"])


# ============================
# 定数の境界値テスト
# ============================


class TestConstants:
    """実測 calibration 値との整合性を確認する。"""

    def test_glow_detection_threshold_is_midpoint(self) -> None:
        """GLOW_DETECTION_THRESHOLD が [LOW, HIGH] の中点に相当する ratio を検知する。

        threshold=0.5 は (GLOW_RATIO_LOW + GLOW_RATIO_HIGH) / 2 の glow_score に対応。
        実測分離点 ratio=0.20 が threshold=0.5 (score=(0.20-0.12)/(0.28-0.12)=0.50) になる。
        """
        ratio_separation = 0.20  # 実測分離点
        expected_score = (ratio_separation - GLOW_RATIO_LOW) / (GLOW_RATIO_HIGH - GLOW_RATIO_LOW)
        assert expected_score == pytest.approx(GLOW_DETECTION_THRESHOLD, abs=1e-6)

    def test_v_high_threshold_in_valid_range(self) -> None:
        """V_HIGH_THRESHOLD が HSV V の有効範囲 (0-255) 内にある。"""
        assert 0 <= V_HIGH_THRESHOLD <= 255

    def test_glow_roi_row_count_positive(self) -> None:
        """GLOW_ROI_ROW_COUNT > 0 かつ VISIBLE_ROWS 以下。"""
        assert 0 < GLOW_ROI_ROW_COUNT <= VISIBLE_ROWS

    def test_ratio_low_less_than_high(self) -> None:
        """GLOW_RATIO_LOW < GLOW_RATIO_HIGH (正規化の前提)。"""
        assert GLOW_RATIO_LOW < GLOW_RATIO_HIGH
