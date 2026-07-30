"""Phase G C-1 / W-α: RecognitionPipeline の prob_board publish 経路テスト。

実画像ではなく stub ImageReader / stub MatchStateDetector で pipeline を組み、
SideResult.prob_board が STABLE 確定時に埋まることを検証する。
さらに hidden_row_inferrer の経路 (TSUMO_FALL → STABLE 着地時に隠し段に
non-zero 確率が乗るケース) を直接 infer_hidden_row でユニットテストする。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.board_state_machine import BoardState
from src.hidden_row_inferrer import infer_hidden_row
from src.old.indicators import IndicatorCalculator
from src.match_state import MatchState
from src.probabilistic_board import ProbabilisticBoard
from src.recognition_pipeline import RecognitionPipeline


# ============================
# stub helpers (test_recognition_pipeline と同じ構造)
# ============================


@dataclass
class _StubMatchResult:
    state: MatchState
    bg_value: float = 100.0
    bg_saturation: float = 50.0
    samples: int = 1


class _StubMatchDetector:
    """常に IN_MATCH を返す MatchStateDetector スタブ."""

    def __init__(self, in_match: bool = True) -> None:
        self._in_match = in_match

    def detect(self, frame: np.ndarray) -> _StubMatchResult:
        return _StubMatchResult(
            state=MatchState.IN_MATCH if self._in_match
            else MatchState.NOT_IN_MATCH,
        )


class _StubImageReader:
    """指定固定 board を返す ImageReader スタブ."""

    def __init__(self, p1: Board, p2: Board) -> None:
        self._p1 = p1
        self._p2 = p2

    def read_both_boards(
        self, frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False,
        skip_tier1_2p: bool = False,
        # 修正2 (2026-07-30): 実 ImageReader.read_both_boards の telop_result
        # 追加 (optional 引数) に追従。スタブでは使わないため受け取るだけ。
        telop_result: object | None = None,
    ) -> tuple[Board, Board]:
        return self._p1.copy(), self._p2.copy()


def _empty_board() -> Board:
    return Board()


def _board_with_red(row: int, col: int) -> Board:
    b = Board()
    b.set(row, col, COLOR_RED)
    return b


def _make_pipe(
    p1: Board, p2: Board, in_match: bool = True, stable_n: int = 2,
) -> RecognitionPipeline:
    reader = _StubImageReader(p1, p2)
    detector = _StubMatchDetector(in_match=in_match)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
    )


def _dummy_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# Test 1: STABLE 遷移で prob_board が publish される
# ============================


def test_stable_publishes_prob_board_fallback() -> None:
    """初回 STABLE 確定 (TSUMO_FALL 経路を通っていない) 時、
    prob_board は ProbabilisticBoard.from_board(confirmed_board) で
    フォールバック生成されている。"""
    p1 = _board_with_red(12, 0)
    p2 = _empty_board()
    pipe = _make_pipe(p1, p2, stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    res = pipe.update(1, 0.05, _dummy_frame())
    assert res.p1.state == BoardState.STABLE
    assert res.p1.prob_board is not None
    # confirmed_board と同じ最尤盤面が出るはず
    mle = res.p1.prob_board.to_max_likelihood_board()
    assert mle == res.p1.confirmed_board


def test_menu_state_has_no_prob_board() -> None:
    """MENU (試合外) では prob_board は None."""
    pipe = _make_pipe(
        _board_with_red(12, 0), _empty_board(),
        in_match=False, stable_n=2,
    )
    res = pipe.update(0, 0.0, _dummy_frame())
    assert res.p1.state == BoardState.MENU
    assert res.p1.prob_board is None
    assert res.p2.prob_board is None


# ============================
# Test 2: prob_board=None 時の compute_all_probabilistic フォールバック
# ============================


def test_compute_all_probabilistic_works_with_from_board() -> None:
    """prob_board=None → ProbabilisticBoard.from_board(confirmed_board)
    でフォールバック → compute_all_probabilistic が問題なく動くこと."""
    confirmed = _board_with_red(12, 0)
    pb = ProbabilisticBoard.from_board(confirmed)
    calc = IndicatorCalculator()
    indicator_set = calc.compute_all_probabilistic(
        pb, n_samples=3,
    )
    # 16 + extra 系 indicator が結果セットに揃う
    assert indicator_set is not None
    assert len(indicator_set.results) > 0


# ============================
# Test 3: infer_hidden_row 経由で隠し段に non-zero 確率が乗る
# ============================


def test_infer_hidden_row_distributes_to_hidden_row() -> None:
    """1 セルしか観測できなかったケース → 隠し段の同列に missing color の
    non-zero 確率が乗る (W-α の中核挙動)."""
    prev = Board()
    cur = Board()
    # row=1 (HIDDEN_ROWS) の col=2 に RED 1 つだけ → ペアの bot が隠し段
    cur.set(1, 2, COLOR_RED)
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.skipped_reason is None
    assert res.cells_with_distribution > 0
    # 列 2 の row 0 で BLUE 確率が non-zero
    cell = pb.cell(0, 2)
    assert cell.get(COLOR_BLUE) > 0.0
    # 別列 (例: col 0) は EMPTY 確定
    assert pb.cell(0, 0).get(COLOR_EMPTY) >= 0.95


def test_infer_hidden_row_two_new_cells_clears_hidden() -> None:
    """通常落下 (2 セル新規) → 隠し段 EMPTY 確定."""
    prev = Board()
    cur = Board()
    cur.set(11, 2, COLOR_RED)
    cur.set(12, 2, COLOR_BLUE)
    pb, res = infer_hidden_row(prev, cur, (COLOR_RED, COLOR_BLUE))
    assert res.skipped_reason is None
    assert res.cells_added_to_hidden > 0
    for c in range(BOARD_COLS):
        cell = pb.cell(0, c)
        assert cell.get(COLOR_EMPTY) >= 0.95


# ============================
# Test 4: NON-STABLE state では prob_board は None
# ============================


def test_non_stable_state_has_none_prob_board() -> None:
    """STABLE に到達する前 (pending 中) は prob_board=None."""
    p1 = _board_with_red(12, 0)
    p2 = _empty_board()
    pipe = _make_pipe(p1, p2, stable_n=5)
    res = pipe.update(0, 0.0, _dummy_frame())
    # 1 frame だけだと STABLE に届かない (stable_n=5)
    assert res.p1.state != BoardState.STABLE
    assert res.p1.prob_board is None
