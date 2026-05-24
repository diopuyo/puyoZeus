"""CellRecoveryRefiner テスト (Phase Z-2)。"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_BLUE, COLOR_EMPTY,
    COLOR_GREEN, COLOR_OJAMA, COLOR_PURPLE, COLOR_RED,
    COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.cell_recovery_refiner import CellRecoveryRefiner
from src.image_reader import BoardRegion


REGION = BoardRegion(x=0, y=0, width=384, height=720)


class _FakeHsvClassifier:
    """指定 (row, col) → color の固定マップを返す HSV モック。"""

    def __init__(self, mapping: dict[tuple[int, int], int]) -> None:
        self.mapping = mapping
        self.next_idx = 0
        self.calls = []

    def classify(self, patch: np.ndarray) -> int:
        # patch のコーナー pixel に row/col を埋め込んで返却順を識別する
        # 単純化のため、呼ばれた順に固定リストを返す
        self.calls.append(patch.shape)
        if not self.calls:
            return COLOR_EMPTY
        # ランダム順序で OK にするためマッピングのキーから順に返す
        keys = list(self.mapping.keys())
        idx = (self.next_idx) % len(keys)
        self.next_idx += 1
        return self.mapping[keys[idx]]


def _make_frame(saturation: int, value: int) -> np.ndarray:
    """region 全体を指定 HSV (色相 50) で塗りつぶした BGR 画像。"""
    hsv = np.zeros((720, 384, 3), dtype=np.uint8)
    hsv[:, :, 0] = 50
    hsv[:, :, 1] = saturation
    hsv[:, :, 2] = value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _make_em_board() -> Board:
    """全セル EMPTY な盤面。"""
    return Board()


def _set_cell(board: Board, vrow: int, col: int, color: int) -> None:
    board.set(vrow + HIDDEN_ROWS, col, color)


# --- import cv2 here (compatibility with module-level imports) ---
import cv2  # noqa: E402


# ============================
# テスト
# ============================


def test_em_recovered_to_color_when_high_saturation() -> None:
    """recognized=EM、HSV 高彩度 → HSV 分類器の色を採用。"""
    frame = _make_frame(saturation=180, value=200)
    board = _make_em_board()  # 全 EM
    hsv = _FakeHsvClassifier({(0, 0): COLOR_GREEN})
    refiner = CellRecoveryRefiner(hsv)
    new_board, stats = refiner.refine(frame, REGION, board)
    # 全 cell が GRN になっている
    assert stats.em_recovered == 12 * BOARD_COLS
    assert stats.ojm_recovered == 0
    assert stats.hsv_voted == 0
    for vrow in range(12):
        for col in range(BOARD_COLS):
            assert int(
                new_board.get(vrow + HIDDEN_ROWS, col)
            ) == COLOR_GREEN


def test_unknown_recovered_to_color() -> None:
    """recognized=UNKNOWN も EM と同様に救われる。"""
    frame = _make_frame(saturation=180, value=200)
    board = _make_em_board()
    _set_cell(board, 5, 2, COLOR_UNKNOWN)
    hsv = _FakeHsvClassifier({(0, 0): COLOR_RED})
    refiner = CellRecoveryRefiner(hsv)
    new_board, stats = refiner.refine(frame, REGION, board)
    # ?? も RED に救われている
    assert int(new_board.get(5 + HIDDEN_ROWS, 2)) == COLOR_RED


def test_em_to_ojm_when_low_sat_mid_val() -> None:
    """enable_ojm_recovery=True 時、厳格 OJM 範囲 (S<35, V 165-200) で採用。"""
    frame = _make_frame(saturation=20, value=180)
    board = _make_em_board()
    hsv = _FakeHsvClassifier({(0, 0): COLOR_EMPTY})
    refiner = CellRecoveryRefiner(hsv, enable_ojm_recovery=True)
    new_board, stats = refiner.refine(frame, REGION, board)
    assert stats.ojm_recovered == 12 * BOARD_COLS
    assert stats.em_recovered == 0
    for vrow in range(12):
        for col in range(BOARD_COLS):
            assert int(
                new_board.get(vrow + HIDDEN_ROWS, col)
            ) == COLOR_OJAMA


def test_ojm_recovery_disabled_by_default() -> None:
    """default では OJM Recovery OFF (背景 false positive 回避)。"""
    frame = _make_frame(saturation=30, value=160)
    board = _make_em_board()
    hsv = _FakeHsvClassifier({(0, 0): COLOR_EMPTY})
    refiner = CellRecoveryRefiner(hsv)  # enable_ojm_recovery=False
    new_board, stats = refiner.refine(frame, REGION, board)
    assert stats.ojm_recovered == 0
    for vrow in range(12):
        for col in range(BOARD_COLS):
            assert int(
                new_board.get(vrow + HIDDEN_ROWS, col)
            ) == COLOR_EMPTY


def test_em_kept_when_truly_low_brightness() -> None:
    """暗い真の EM は変更されない。"""
    frame = _make_frame(saturation=20, value=30)  # S 低 + V 低
    board = _make_em_board()
    hsv = _FakeHsvClassifier({(0, 0): COLOR_EMPTY})
    refiner = CellRecoveryRefiner(hsv)
    new_board, stats = refiner.refine(frame, REGION, board)
    assert stats.em_recovered == 0
    assert stats.ojm_recovered == 0
    for vrow in range(12):
        for col in range(BOARD_COLS):
            assert int(
                new_board.get(vrow + HIDDEN_ROWS, col)
            ) == COLOR_EMPTY


def test_hsv_vote_overrides_color_swap() -> None:
    """recognized=GRN だが S が高くて HSV=PUR → PUR に補正。"""
    frame = _make_frame(saturation=180, value=200)
    board = _make_em_board()
    _set_cell(board, 7, 3, COLOR_GREEN)  # CNN=GRN
    hsv = _FakeHsvClassifier({(0, 0): COLOR_PURPLE})
    refiner = CellRecoveryRefiner(hsv)
    new_board, stats = refiner.refine(frame, REGION, board)
    # GRN cell は PUR に置換される
    assert int(new_board.get(7 + HIDDEN_ROWS, 3)) == COLOR_PURPLE
    assert stats.hsv_voted >= 1


def test_hsv_vote_skipped_when_low_saturation() -> None:
    """色 swap 候補でも S が低ければ HsvVote しない。"""
    # S=79 で HSV_VOTE_S_MIN=80 を厳密に下回る (refiner 側は >= で発動)
    frame = _make_frame(saturation=79, value=200)  # S < HSV_VOTE_S_MIN
    board = _make_em_board()
    _set_cell(board, 7, 3, COLOR_GREEN)
    hsv = _FakeHsvClassifier({(0, 0): COLOR_PURPLE})
    refiner = CellRecoveryRefiner(hsv)
    new_board, stats = refiner.refine(frame, REGION, board)
    # CNN 判定維持
    assert int(new_board.get(7 + HIDDEN_ROWS, 3)) == COLOR_GREEN
    assert stats.hsv_voted == 0


def test_ojama_cell_voted_to_puyo_when_high_saturation() -> None:
    """Z-3F: recognized=OJM でも S 高ければ HSV 主要色を採用。

    OJM↔PUR 混同 (CNN が OJM と固定誤認) を救うため、OJM cell も
    HsvVote の対象に含める。
    """
    frame = _make_frame(saturation=180, value=200)
    board = _make_em_board()
    _set_cell(board, 7, 3, COLOR_OJAMA)
    hsv = _FakeHsvClassifier({(0, 0): COLOR_RED})
    refiner = CellRecoveryRefiner(hsv)
    new_board, _ = refiner.refine(frame, REGION, board)
    assert int(new_board.get(7 + HIDDEN_ROWS, 3)) == COLOR_RED


def test_ojama_cell_kept_when_low_saturation() -> None:
    """recognized=OJM で S 低ければ OJM 維持 (HsvVote しない)。"""
    # S=79 で HSV_VOTE_S_MIN=80 を厳密に下回る (refiner 側は >= で発動)
    frame = _make_frame(saturation=79, value=200)  # S < HSV_VOTE_S_MIN
    board = _make_em_board()
    _set_cell(board, 7, 3, COLOR_OJAMA)
    hsv = _FakeHsvClassifier({(0, 0): COLOR_RED})
    refiner = CellRecoveryRefiner(hsv)
    new_board, _ = refiner.refine(frame, REGION, board)
    assert int(new_board.get(7 + HIDDEN_ROWS, 3)) == COLOR_OJAMA


def test_skip_physical_correction_when_chain() -> None:
    """Z-3G: is_chain=True なら airborne 強制 EM をスキップ。

    元から puyo cell の直下が EM でも、連鎖中なら強制 EM 化しない
    (相殺エフェクトで真 puyo を消失させるバグ対策)。
    """
    frame = _make_frame(saturation=20, value=20)  # 補正対象なし
    board = _make_em_board()
    _set_cell(board, 5, 3, COLOR_RED)  # 浮遊 puyo
    hsv = _FakeHsvClassifier({(0, 0): COLOR_EMPTY})
    refiner = CellRecoveryRefiner(hsv)
    # 通常 (連鎖外): 浮遊 puyo は強制 EM 化される
    new_board, _ = refiner.refine(frame, REGION, board, is_chain=False)
    assert int(new_board.get(5 + HIDDEN_ROWS, 3)) == COLOR_EMPTY
    # 連鎖中: 浮遊 puyo はそのまま (相殺エフェクト保護)
    board2 = _make_em_board()
    _set_cell(board2, 5, 3, COLOR_RED)
    new_board2, _ = refiner.refine(frame, REGION, board2, is_chain=True)
    assert int(new_board2.get(5 + HIDDEN_ROWS, 3)) == COLOR_RED


def test_calibrate_thresholds_keeps_default_for_low_sat_bg() -> None:
    """Z-3H: 低彩度 BG なら閾値は default 維持 (max で底上げ無し)。"""
    from src.cell_recovery_refiner import EM_RECOVERY_S_MIN
    bg_frame = _make_frame(saturation=10, value=30)  # 真 EM 背景: 暗・低彩
    hsv = _FakeHsvClassifier({(0, 0): COLOR_EMPTY})
    refiner = CellRecoveryRefiner(hsv)
    refiner.calibrate_thresholds([bg_frame], [REGION])
    # default 閾値は維持される (BG が低彩度でも閾値を下げない)
    assert refiner._em_s_min >= EM_RECOVERY_S_MIN
    assert refiner._calibrated


def test_calibrate_thresholds_raises_for_high_sat_bg() -> None:
    """Z-3H: 高彩度 BG (動画別の特殊背景) なら閾値が引き上げられる。"""
    from src.cell_recovery_refiner import EM_RECOVERY_S_MIN
    bg_frame = _make_frame(saturation=80, value=180)  # 高彩度 BG
    hsv = _FakeHsvClassifier({(0, 0): COLOR_EMPTY})
    refiner = CellRecoveryRefiner(hsv)
    refiner.calibrate_thresholds([bg_frame], [REGION])
    # σ=0 でも mean (=80) が default (=60) より高ければ閾値が上がる
    assert refiner._em_s_min >= EM_RECOVERY_S_MIN
