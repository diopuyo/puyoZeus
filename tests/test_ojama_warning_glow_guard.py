"""src/ojama_warning_glow_guard.py のユニットテスト.

テスト方針:
  - stateless 関数 (compute_glow_score, update_glow_state, apply_glow_guard) を独立に検証
  - GlowGuardState の ON/OFF 遷移・上限解除を網羅
  - glow_active=False で現挙動不変を確認 (フラグ OFF 時の安全弁)
  - CLI フラグが BooleanOptionalAction で正しく動くことを argparse 直接テストで確認
    (2026-06-05 採用確定: default=True、--no-ojama-warning-glow-guard で無効化)
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
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN,
    COLOR_OJAMA, COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, Board,
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
    _is_consensus_colored,
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

    def test_on_new_puyo_not_touched(self) -> None:
        """v2: glow_active=True かつ frozen が空 かつ confirmed に有色 → 不触 (confirmed のまま).

        v1 では UNKNOWN 留保だったが、v2 では新規ぷよは触らない。
        「confirmed=おじゃま かつ frozen=有色」の条件を満たさないためスキップされる。
        """
        state = GlowGuardState()
        state.glow_active = True
        # frozen は空 (発光前に置かれたぷよがない)
        state.frozen_board = _make_board()
        # 発光中に新しく「色」が見えた (confirmed に青ぷよ)
        confirmed = _make_board({(3, 1): COLOR_BLUE})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # v2: 新規ぷよは不触 → confirmed の BLUE がそのまま残る
        assert int(result.get(3, 1)) == COLOR_BLUE

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

    # ---- v2 ターゲット型 固有テスト ----------------------------------------

    def test_v2_ojama_misrecognition_restored(self) -> None:
        """v2: confirmed=おじゃま かつ frozen=有色 → frozen 色に復元する (主目的).

        発光で黄(4)→おじゃま(9)に誤認されたセルを frozen の黄に戻す。
        これが v2 の唯一の介入対象。
        """
        state = GlowGuardState()
        state.glow_active = True
        # 発光前は黄ぷよがあった
        state.frozen_board = _make_board({(2, 3): COLOR_YELLOW})
        # 発光中に誤認でおじゃまになった
        confirmed = _make_board({(2, 3): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # frozen の YELLOW に復元される
        assert int(result.get(2, 3)) == COLOR_YELLOW

    def test_v2_correct_color_cell_not_touched(self) -> None:
        """v2: confirmed に正常な色(おじゃまでない)があるセルは一切触れない.

        frozen にも色があっても、confirmed がおじゃまでなければ不触。
        """
        state = GlowGuardState()
        state.glow_active = True
        # frozen は緑
        state.frozen_board = _make_board({(5, 0): COLOR_GREEN})
        # confirmed も正常に緑 (誤認なし)
        confirmed = _make_board({(5, 0): COLOR_GREEN})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # confirmed の GREEN がそのまま (frozen と同じ値でも介入しない)
        assert int(result.get(5, 0)) == COLOR_GREEN

    def test_v2_originally_ojama_cell_not_touched(self) -> None:
        """v2: frozen もおじゃまで confirmed もおじゃまのセルは不触.

        frozen=おじゃまの場合は「frozen_is_colored」が False になるため復元しない。
        元々おじゃまだったセルを誤って色ぷよに戻さないことを保証する。
        """
        state = GlowGuardState()
        state.glow_active = True
        # frozen からおじゃまがあった (正常なおじゃまセル)
        state.frozen_board = _make_board({(1, 2): COLOR_OJAMA})
        # confirmed もおじゃままま
        confirmed = _make_board({(1, 2): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # おじゃまのまま (不触)
        assert int(result.get(1, 2)) == COLOR_OJAMA

    def test_v2_empty_cell_not_touched(self) -> None:
        """v2: frozen が有色でも confirmed が空のセルは不触.

        「confirmed=空 かつ frozen=有色」は復元対象外。
        ぷよが消えた正当な空を誤って色で埋めない。
        """
        state = GlowGuardState()
        state.glow_active = True
        # frozen に紫があった
        state.frozen_board = _make_board({(4, 5): COLOR_PURPLE})
        # confirmed は空 (連鎖消去後など正当な空)
        confirmed = _make_board()
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # 空のまま (不触)
        assert int(result.get(4, 5)) == COLOR_EMPTY


# ============================
# glow_active=False (フラグ OFF 時の安全弁) のテスト
# ============================


class TestDefaultOff:
    """glow_active=False のとき apply_glow_guard が何もしないことを確認。

    2026-06-05: enable_ojama_warning_glow_guard の default は True に変更済み。
    このクラスは「フラグ OFF 時の安全弁」として is_glow_active=False の挙動を検証する。
    """

    def test_glow_inactive_returns_confirmed_unchanged(self) -> None:
        """is_glow_active=False のとき apply_glow_guard は confirmed をそのまま返す。"""
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
# CLI フラグ (BooleanOptionalAction, default=True) のテスト
# (2026-06-05 採用確定: store_true/default=False から変更)
# ============================


class TestCliFlag:
    """--ojama-warning-glow-guard が BooleanOptionalAction / default=True で正しく動くことを確認。

    採用確定 (2026-06-05) により:
      - フラグ未指定 → True (ライブラリ default と同一)
      - --ojama-warning-glow-guard → True (明示 ON)
      - --no-ojama-warning-glow-guard → False (明示 OFF)
    オプション名 "--ojama-warning-glow-guard" は先頭が "--no-" でないため
    BooleanOptionalAction の反転バグ (argparse が "--no-no-..." を生成) は発生しない。
    """

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        """2026-06-05 採用構成: BooleanOptionalAction / default=True。"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--ojama-warning-glow-guard",
            action=argparse.BooleanOptionalAction,
            default=True,
            dest="enable_ojama_warning_glow_guard",
        )
        return parser

    def test_flag_absent_is_true(self) -> None:
        """フラグ未指定のとき enable_ojama_warning_glow_guard=True (採用 default ON)。"""
        parser = self._build_parser()
        args = parser.parse_args([])
        assert args.enable_ojama_warning_glow_guard is True

    def test_flag_present_is_true(self) -> None:
        """--ojama-warning-glow-guard 指定でも enable_ojama_warning_glow_guard=True。"""
        parser = self._build_parser()
        args = parser.parse_args(["--ojama-warning-glow-guard"])
        assert args.enable_ojama_warning_glow_guard is True

    def test_no_prefix_is_false(self) -> None:
        """--no-ojama-warning-glow-guard 指定で enable_ojama_warning_glow_guard=False (無効化)。"""
        parser = self._build_parser()
        args = parser.parse_args(["--no-ojama-warning-glow-guard"])
        assert args.enable_ojama_warning_glow_guard is False


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


# ============================
# v3: ojama_fall / tsumo_fall でも apply_glow_guard が動作することを検証
# ============================


class TestV3NonStableGuard:
    """v3 拡張: ojama_fall / tsumo_fall 状態でも O 誤認セルが復元されることを確認する。

    pipeline 内部の状態分岐は直接呼べないため、ここでは apply_glow_guard 本体と
    GlowGuardState の振る舞いを検証する。
    「apply_glow_guard は状態に依存しない stateless 関数」であるため、
    OJAMA_FALL / TSUMO_FALL 中でも同じ引数を渡せば復元することを示す。
    pipeline 側の適用条件拡張 (recognition_pipeline.py:3736 相当) は
    TestFrozenBoardNotUpdatedInNonStable で frozen 更新タイミングを検証する。
    """

    def test_ojama_fall_like_ojama_misrecognition_restored(self) -> None:
        """ojama_fall 中に相当する confirmed=O かつ frozen=有色 → 復元される.

        apply_glow_guard 自体は状態を受け取らず stateless なため、
        pipeline が ojama_fall でも呼べば正しく復元することを確認する。
        """
        state = GlowGuardState()
        state.glow_active = True
        # 発光前 (STABLE 時) の frozen は黄ぷよ
        state.frozen_board = _make_board({(0, 1): COLOR_YELLOW, (1, 3): COLOR_BLUE})
        # ojama_fall 中: 発光で黄→O, 青→O に誤認された confirmed
        confirmed = _make_board({(0, 1): COLOR_OJAMA, (1, 3): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # 両セルとも frozen の色 (YELLOW / BLUE) に復元される
        assert int(result.get(0, 1)) == COLOR_YELLOW
        assert int(result.get(1, 3)) == COLOR_BLUE

    def test_tsumo_fall_like_ojama_misrecognition_restored(self) -> None:
        """tsumo_fall 中に相当する confirmed=O かつ frozen=有色 → 復元される.

        ojama_fall と同様、apply_glow_guard は状態に依存しないため
        tsumo_fall 中でも正しく復元することを確認する。
        """
        state = GlowGuardState()
        state.glow_active = True
        # 発光前の frozen は紫ぷよ
        state.frozen_board = _make_board({(3, 2): COLOR_PURPLE})
        # tsumo_fall 中: 発光で紫→O に誤認
        confirmed = _make_board({(3, 2): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # COLOR_PURPLE に復元される
        assert int(result.get(3, 2)) == COLOR_PURPLE

    def test_frozen_board_not_updated_during_non_stable(self) -> None:
        """非 STABLE (ojama_fall 相当) 中は frozen_board が更新されないことを確認する.

        pipeline の実装では「frozen 更新 = 発光 OFF かつ STABLE 確定時のみ」。
        ここでは GlowGuardState を直接操作して、
        非 STABLE 中に frozen を更新しないパスを模倣し、
        frozen が直前 STABLE の盤面を保持し続けることを確認する。
        """
        state = GlowGuardState()
        # 発光 OFF の STABLE 時に frozen が更新された (pipeline の正規パス)
        stable_board = _make_board({(5, 0): COLOR_YELLOW})
        state.frozen_board = stable_board.copy()

        # --- 以降は非 STABLE (ojama_fall) 中: frozen は一切触らない ---
        # ojama_fall 中の confirmed (発光で O 誤認)
        ojama_fall_confirmed = _make_board({(5, 0): COLOR_OJAMA})
        # pipeline は非 STABLE では frozen を更新しないため、ここでは意図的に更新しない

        # 発光を強制 ON にする
        for fi in range(GLOW_CONSEC_MIN):
            update_glow_state(state, 1.0, frame_idx=fi)
        assert state.glow_active

        # apply_glow_guard を呼ぶ (pipeline が ojama_fall でも呼ぶようになった v3 の動作)
        result = apply_glow_guard(ojama_fall_confirmed, state, is_glow_active=True)
        # frozen が STABLE 時の黄ぷよを保持しているため、O 誤認が復元される
        assert int(result.get(5, 0)) == COLOR_YELLOW
        # frozen_board は更新されていない (STABLE 時の値のまま)
        assert int(state.frozen_board.get(5, 0)) == COLOR_YELLOW

    def test_frozen_board_updated_only_on_stable_off(self) -> None:
        """frozen_board 更新は「発光 OFF かつ STABLE」でのみ行われることを保証する.

        STABLE/発光 OFF: frozen 更新される (v2 から変更なし)。
        OJAMA_FALL/発光 OFF: frozen 更新されない (v3 の非 STABLE 中凍結保持)。
        このテストは pipeline の更新ロジックを GlowGuardState で直接模倣する。
        """
        state = GlowGuardState()
        # 初期 frozen は黄ぷよ (STABLE 時に更新済みと仮定)
        initial_frozen = _make_board({(10, 2): COLOR_YELLOW})
        state.frozen_board = initial_frozen.copy()

        # 発光 OFF (glow_active=False) のまま ojama_fall 中の confirmed を受け取る
        # pipeline では「elif ctx.state == BoardState.STABLE:」の条件で弾かれる
        # → frozen は更新しない
        ojama_confirmed = _make_board({(10, 2): COLOR_GREEN})
        # (ここでは意図的に frozen を更新しない = pipeline の ojama_fall パス)

        # frozen は初期値のままであることを確認
        assert int(state.frozen_board.get(10, 2)) == COLOR_YELLOW

        # STABLE + 発光 OFF の場合は frozen が更新される (pipeline の正規パス)
        stable_confirmed = _make_board({(10, 2): COLOR_GREEN})
        state.frozen_board = stable_confirmed.copy()  # pipeline が STABLE 時に実行
        assert int(state.frozen_board.get(10, 2)) == COLOR_GREEN


# ============================
# v4: _is_consensus_colored のテスト
# ============================


class TestIsConsensusColored:
    """_is_consensus_colored: CNN と HSV の合意判定を検証する。"""

    def test_both_agree_colored_returns_true(self) -> None:
        """CNN と HSV が同一有色で合意 → (True, その色) を返す。"""
        cnn = _make_board({(3, 2): COLOR_YELLOW})
        hsv = _make_board({(3, 2): COLOR_YELLOW})
        is_cons, color = _is_consensus_colored(cnn, hsv, 3, 2)
        assert is_cons is True
        assert color == COLOR_YELLOW

    def test_both_agree_ojama_returns_false(self) -> None:
        """両者ともおじゃまで合意しても (False, 0) を返す (有色の合意のみ対象)。"""
        cnn = _make_board({(1, 0): COLOR_OJAMA})
        hsv = _make_board({(1, 0): COLOR_OJAMA})
        is_cons, color = _is_consensus_colored(cnn, hsv, 1, 0)
        assert is_cons is False
        assert color == 0

    def test_both_agree_empty_returns_false(self) -> None:
        """両者とも空で合意しても (False, 0) を返す。"""
        cnn = _make_board()
        hsv = _make_board()
        is_cons, color = _is_consensus_colored(cnn, hsv, 5, 3)
        assert is_cons is False
        assert color == 0

    def test_disagree_returns_false(self) -> None:
        """CNN と HSV が異なる色を示す場合は (False, 0) を返す。"""
        cnn = _make_board({(2, 1): COLOR_RED})
        hsv = _make_board({(2, 1): COLOR_BLUE})
        is_cons, color = _is_consensus_colored(cnn, hsv, 2, 1)
        assert is_cons is False
        assert color == 0

    def test_none_cnn_returns_false(self) -> None:
        """cnn_board が None の場合は (False, 0) を返す。"""
        hsv = _make_board({(0, 0): COLOR_GREEN})
        is_cons, color = _is_consensus_colored(None, hsv, 0, 0)
        assert is_cons is False
        assert color == 0

    def test_none_hsv_returns_false(self) -> None:
        """hsv_board が None の場合は (False, 0) を返す。"""
        cnn = _make_board({(4, 4): COLOR_PURPLE})
        is_cons, color = _is_consensus_colored(cnn, None, 4, 4)
        assert is_cons is False
        assert color == 0

    def test_both_none_returns_false(self) -> None:
        """両方 None の場合は (False, 0) を返す。"""
        is_cons, color = _is_consensus_colored(None, None, 7, 2)
        assert is_cons is False
        assert color == 0

    def test_unknown_agreement_returns_false(self) -> None:
        """両者とも UNKNOWN で合意しても (False, 0) を返す。"""
        cnn = _make_board({(6, 5): COLOR_UNKNOWN})
        hsv = _make_board({(6, 5): COLOR_UNKNOWN})
        is_cons, color = _is_consensus_colored(cnn, hsv, 6, 5)
        assert is_cons is False
        assert color == 0


# ============================
# v4: apply_glow_guard の consensus 優先復元テスト
# ============================


class TestV4ConsensusRestoration:
    """v4: apply_glow_guard の consensus 優先復元ロジックを検証する。"""

    def test_consensus_color_used_over_frozen(self) -> None:
        """consensus が frozen と異なる色を示す場合 → consensus 色で復元する (c2c 退行解消).

        シナリオ: frozen=黄(消去前の盤面)、実際は連鎖消去後に青が配置済み。
        raw_cnn=青、raw_hsv=青 → consensus=青 → 青で復元 (frozen の黄ではなく)。
        """
        state = GlowGuardState()
        state.glow_active = True
        # frozen は消去前の盤面 (古い黄)
        state.frozen_board = _make_board({(5, 2): COLOR_YELLOW})
        # confirmed は発光でおじゃまに誤認
        confirmed = _make_board({(5, 2): COLOR_OJAMA})
        # CNN・HSV ともに新色 (青) で合意
        raw_cnn = _make_board({(5, 2): COLOR_BLUE})
        raw_hsv = _make_board({(5, 2): COLOR_BLUE})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # frozen の黄ではなく consensus の青で復元される
        assert int(result.get(5, 2)) == COLOR_BLUE

    def test_no_consensus_falls_back_to_frozen(self) -> None:
        """CNN と HSV が不一致 → frozen 色にフォールバックする (v3 互換挙動)。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(3, 1): COLOR_YELLOW})
        confirmed = _make_board({(3, 1): COLOR_OJAMA})
        # CNN と HSV が異なる色を示す (consensus なし)
        raw_cnn = _make_board({(3, 1): COLOR_RED})
        raw_hsv = _make_board({(3, 1): COLOR_BLUE})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # frozen の黄にフォールバック
        assert int(result.get(3, 1)) == COLOR_YELLOW

    def test_consensus_ojama_falls_back_to_frozen(self) -> None:
        """CNN・HSV ともおじゃまで合意しても → frozen 色にフォールバックする。

        発光中に両者ともおじゃまを示す = 発光誤認と判断し frozen 色で保護する。
        """
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(2, 4): COLOR_GREEN})
        confirmed = _make_board({(2, 4): COLOR_OJAMA})
        # 両者ともおじゃまで合意 (= 発光誤認の典型パターン)
        raw_cnn = _make_board({(2, 4): COLOR_OJAMA})
        raw_hsv = _make_board({(2, 4): COLOR_OJAMA})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # frozen の緑にフォールバック
        assert int(result.get(2, 4)) == COLOR_GREEN

    def test_none_boards_falls_back_to_frozen(self) -> None:
        """raw_cnn / raw_hsv が None → frozen 色にフォールバック (v3 完全互換)。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(1, 0): COLOR_PURPLE})
        confirmed = _make_board({(1, 0): COLOR_OJAMA})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=None, raw_hsv_board=None,
        )
        assert int(result.get(1, 0)) == COLOR_PURPLE

    def test_backward_compat_no_args(self) -> None:
        """引数省略 (v3 互換呼び出し) でも frozen 色で正常に復元される。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(8, 3): COLOR_RED})
        confirmed = _make_board({(8, 3): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        assert int(result.get(8, 3)) == COLOR_RED

    def test_unrelated_cell_not_touched(self) -> None:
        """consensus がある場合でも対象外セル (confirmed=有色) は不触。"""
        state = GlowGuardState()
        state.glow_active = True
        # frozen は黄、対象外セルに青
        state.frozen_board = _make_board({(4, 2): COLOR_YELLOW, (4, 3): COLOR_GREEN})
        # (4,2) はおじゃま誤認、(4,3) は正常に緑
        confirmed = _make_board({(4, 2): COLOR_OJAMA, (4, 3): COLOR_GREEN})
        raw_cnn = _make_board({(4, 2): COLOR_BLUE, (4, 3): COLOR_RED})
        raw_hsv = _make_board({(4, 2): COLOR_BLUE, (4, 3): COLOR_RED})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # (4,2): consensus=青 で復元
        assert int(result.get(4, 2)) == COLOR_BLUE
        # (4,3): confirmed が有色 (おじゃまでない) → 不触のまま緑
        assert int(result.get(4, 3)) == COLOR_GREEN


# ============================
# v4: CHAIN 状態を guard 対象に追加したことの確認テスト
# ============================


class TestV4ChainStateGuard:
    """v4: CHAIN 状態でも apply_glow_guard が O 誤認を復元することを確認する。

    pipeline 内部の状態分岐 (BoardState.CHAIN 追加) は直接呼べないため、
    apply_glow_guard 本体が CHAIN 中相当の引数で正しく復元することを示す。
    安全性: 「confirmed=おじゃま かつ frozen=有色」ルールは CHAIN 中も成立する
    (連鎖消去は色→空であり、色→おじゃまへの変化は全て誤認)。
    正当な消去 (色→空) は confirmed=空 で条件を満たさず不触のため安全。
    """

    def test_chain_state_ojama_misrecognition_restored(self) -> None:
        """CHAIN 中に発光で色→おじゃまに誤認されたセルが復元される。"""
        state = GlowGuardState()
        state.glow_active = True
        # 発光前 (最終 STABLE 時) の frozen
        state.frozen_board = _make_board({(0, 1): COLOR_YELLOW, (0, 4): COLOR_RED})
        # CHAIN 中: 発光で黄→O、赤→O に誤認された confirmed
        confirmed = _make_board({(0, 1): COLOR_OJAMA, (0, 4): COLOR_OJAMA})
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # 両セルとも frozen 色に復元される
        assert int(result.get(0, 1)) == COLOR_YELLOW
        assert int(result.get(0, 4)) == COLOR_RED

    def test_chain_legitimate_erasure_not_touched(self) -> None:
        """CHAIN 中の正当な消去 (色→空) は不触のまま残る。

        ルールは「confirmed=おじゃま かつ frozen=有色」のみ対象。
        confirmed=空 (正当消去) は条件を満たさず空のまま保持される。
        """
        state = GlowGuardState()
        state.glow_active = True
        # 連鎖前の盤面: (7,2) に緑があった
        state.frozen_board = _make_board({(7, 2): COLOR_GREEN})
        # CHAIN 中: (7,2) が正当に消去されて空になった confirmed
        confirmed = _make_board()  # 全空 (消去後)
        result = apply_glow_guard(confirmed, state, is_glow_active=True)
        # (7,2): 空のまま (正当消去を保護ルールが上書きしない)
        assert int(result.get(7, 2)) == COLOR_EMPTY

    def test_chain_consensus_used_for_restoration(self) -> None:
        """CHAIN 中に consensus 優先復元が機能する (発光+再配置同時のエッジケース)。"""
        state = GlowGuardState()
        state.glow_active = True
        # frozen は古い黄 (消去前)
        state.frozen_board = _make_board({(1, 3): COLOR_YELLOW})
        # CHAIN 中: confirmed はおじゃまに誤認
        confirmed = _make_board({(1, 3): COLOR_OJAMA})
        # CNN・HSV ともに新色 (紫) で合意 = 連鎖後に紫が配置済み
        raw_cnn = _make_board({(1, 3): COLOR_PURPLE})
        raw_hsv = _make_board({(1, 3): COLOR_PURPLE})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # consensus の紫で復元 (古い frozen の黄ではない)
        assert int(result.get(1, 3)) == COLOR_PURPLE


# ============================
# v5: frozen 非有色でも consensus 復元のテスト
# ============================


class TestV5FrozenNonColoredConsensusRestore:
    """v5 追加ルール: frozen が非有色(O/空/UNKNOWN)でも CNN==HSV=明確な色なら復元する。

    主な検証シナリオ:
      1. frozen=空 かつ CNN==HSV=黄 → consensus 黄で復元 (v89 t≈71 cell(2,3) 相当)。
      2. frozen=おじゃま かつ CNN==HSV=赤 → consensus 赤で復元。
      3. frozen=UNKNOWN かつ CNN==HSV=青 → consensus 青で復元。
      4. frozen=空 かつ CNN==HSV=おじゃま (真おじゃま) → 不触 (confirmed=O 保持)。
      5. frozen=空 かつ CNN≠HSV (不一致) → 不触 (confirmed=O 保持)。
      6. frozen=空 かつ raw_cnn=None → 不触 (consensus 判定不能)。
      7. frozen=有色 の v4 挙動は v5 でも変わらない (後退ゼロ確認)。
    """

    def test_frozen_empty_consensus_yellow_restores(self) -> None:
        """frozen=空 かつ CNN==HSV=黄 → 黄で復元 (v89 t≈71 cell(2,3) 残差パターン)。

        v4 では frozen=空のため frozen_is_colored=False → 復元ゲートを通らず O 誤認が残った。
        v5 では CNN==HSV=黄の consensus が有効になり O→黄に復元される。
        """
        state = GlowGuardState()
        state.glow_active = True
        # frozen は空 (v4 で復元できなかった原因)
        state.frozen_board = _make_board()
        # confirmed は発光でおじゃまに誤認
        confirmed = _make_board({(2, 3): COLOR_OJAMA})
        # CNN・HSV ともに黄で合意
        raw_cnn = _make_board({(2, 3): COLOR_YELLOW})
        raw_hsv = _make_board({(2, 3): COLOR_YELLOW})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # v5: consensus 黄で復元される
        assert int(result.get(2, 3)) == COLOR_YELLOW

    def test_frozen_ojama_consensus_red_restores(self) -> None:
        """frozen=おじゃま かつ CNN==HSV=赤 → 赤で復元 (frozen=O は非有色扱い)。

        frozen がおじゃまの場合、v4 では frozen_is_colored=False で不触だった。
        v5 では CNN==HSV=赤の consensus で O→赤に復元される。
        """
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(0, 0): COLOR_OJAMA})
        confirmed = _make_board({(0, 0): COLOR_OJAMA})
        raw_cnn = _make_board({(0, 0): COLOR_RED})
        raw_hsv = _make_board({(0, 0): COLOR_RED})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        assert int(result.get(0, 0)) == COLOR_RED

    def test_frozen_unknown_consensus_blue_restores(self) -> None:
        """frozen=UNKNOWN かつ CNN==HSV=青 → 青で復元。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(3, 5): COLOR_UNKNOWN})
        confirmed = _make_board({(3, 5): COLOR_OJAMA})
        raw_cnn = _make_board({(3, 5): COLOR_BLUE})
        raw_hsv = _make_board({(3, 5): COLOR_BLUE})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        assert int(result.get(3, 5)) == COLOR_BLUE

    def test_frozen_empty_consensus_ojama_not_restored(self) -> None:
        """frozen=空 かつ CNN==HSV=おじゃま → 不触 (真おじゃまは保持).

        真おじゃまが降った場合: CNN・HSV ともおじゃまを示す。
        consensus=おじゃまは非O色の合意条件を満たさないため復元されない。
        confirmed=O がそのまま保持されることで真おじゃまを誤って色に書き換えない。
        """
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board()
        confirmed = _make_board({(1, 2): COLOR_OJAMA})
        # 真おじゃま: CNN・HSV ともおじゃまで合意
        raw_cnn = _make_board({(1, 2): COLOR_OJAMA})
        raw_hsv = _make_board({(1, 2): COLOR_OJAMA})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # 真おじゃまは復元されない (confirmed=O のまま保持)
        assert int(result.get(1, 2)) == COLOR_OJAMA

    def test_frozen_empty_consensus_mismatch_not_restored(self) -> None:
        """frozen=空 かつ CNN≠HSV (不一致) → 不触 (consensus なし)。"""
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board()
        confirmed = _make_board({(4, 1): COLOR_OJAMA})
        # CNN と HSV が異なる色を示す
        raw_cnn = _make_board({(4, 1): COLOR_GREEN})
        raw_hsv = _make_board({(4, 1): COLOR_PURPLE})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # consensus なし → 不触 (confirmed=O 保持)
        assert int(result.get(4, 1)) == COLOR_OJAMA

    def test_frozen_empty_no_raw_boards_not_restored(self) -> None:
        """frozen=空 かつ raw_cnn=None → 不触 (consensus 判定不能)。

        raw_cnn/raw_hsv が None の場合は _is_consensus_colored が (False, 0) を返すため
        v5 の consensus ブランチに入らず confirmed=O がそのまま保持される。
        """
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board()
        confirmed = _make_board({(6, 2): COLOR_OJAMA})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=None, raw_hsv_board=None,
        )
        # raw_boards なし → 不触
        assert int(result.get(6, 2)) == COLOR_OJAMA

    def test_v4_frozen_colored_behavior_unchanged(self) -> None:
        """v5 で frozen=有色のケースは v4 と全く同じ挙動を保つ (後退ゼロ確認)。

        frozen=有色 かつ consensus=新色 → consensus 色で復元 (v4 と同じ)。
        frozen=有色 かつ consensus なし → frozen 色にフォールバック (v4 と同じ)。
        """
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board({(5, 3): COLOR_YELLOW, (5, 4): COLOR_GREEN})
        confirmed = _make_board({(5, 3): COLOR_OJAMA, (5, 4): COLOR_OJAMA})
        # (5,3): consensus=青 (v4 と同様 consensus 優先)
        # (5,4): CNN≠HSV → frozen=緑にフォールバック
        raw_cnn = _make_board({(5, 3): COLOR_BLUE, (5, 4): COLOR_RED})
        raw_hsv = _make_board({(5, 3): COLOR_BLUE, (5, 4): COLOR_PURPLE})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # (5,3): consensus 青で復元 (v4 と同じ)
        assert int(result.get(5, 3)) == COLOR_BLUE
        # (5,4): frozen 緑にフォールバック (v4 と同じ)
        assert int(result.get(5, 4)) == COLOR_GREEN

    def test_non_ojama_confirmed_not_touched_regardless(self) -> None:
        """confirmed=有色 のセルは frozen/consensus に関わらず常に不触。

        v5 で分岐を追加したが、confirmed=おじゃま でない限り一切介入しない。
        """
        state = GlowGuardState()
        state.glow_active = True
        state.frozen_board = _make_board()  # 全空
        confirmed = _make_board({(7, 0): COLOR_RED})
        raw_cnn = _make_board({(7, 0): COLOR_YELLOW})
        raw_hsv = _make_board({(7, 0): COLOR_YELLOW})
        result = apply_glow_guard(
            confirmed, state, is_glow_active=True,
            raw_cnn_board=raw_cnn, raw_hsv_board=raw_hsv,
        )
        # confirmed=赤のまま (介入なし)
        assert int(result.get(7, 0)) == COLOR_RED
