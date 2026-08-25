"""W38 根治 (2026-08-25) の配線テスト。

docs/KNOWN_WEAKNESSES.md W38:
    RecognitionPipeline が VideoChainTracker に match_start_sec を渡しておらず、
    ChainEvent.ojama_sent のマージンタイム減衰 (src/scoring.py
    compute_effective_rate、試合開始96秒起点・16秒ごと0.75倍・下限1) が
    「動画の絶対時刻」で計算されていた。320秒を超える長時間動画では
    レートが常に 1 になり、点数がそのままおじゃま個数になっていた
    (実測: t=803.7 の連鎖で 正しい1,579個 vs 壊れた110,540個)。

根治内容:
    - RecognitionPipeline.reset() に optional 引数 match_start_sec を追加
      (backwards compat: 無引数呼び出しは従来通り 0.0)。
    - 試合境界検知 (score リセット境界) の self.reset() 呼び出しで
      境界フレームの time_sec を match_start_sec として渡す。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.chain_detector import VideoChainTracker
from src.match_state import MatchState
from src.recognition_pipeline import RecognitionPipeline


# ============================
# stub helpers (tests/test_recognition_pipeline.py と同パターン)
# ============================


@dataclass
class _StubMatchResult:
    state: MatchState
    bg_value: float = 100.0
    bg_saturation: float = 50.0
    samples: int = 1


class _StubMatchDetector:
    """常に IN_MATCH を返す MatchStateDetector スタブ."""

    def detect(self, frame: np.ndarray) -> _StubMatchResult:
        return _StubMatchResult(state=MatchState.IN_MATCH)


class _StubImageReader:
    """常に空盤面を返す ImageReader スタブ."""

    def read_both_boards(
        self, frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False,
        skip_tier1_2p: bool = False,
        telop_result: object | None = None,
    ) -> tuple[Board, Board]:
        return Board(), Board()


class _StubScoreOcr:
    """常に固定 score を返す ScoreOcr スタブ (ScoreTracker 経由で使用)。

    両サイド score=0 (≤ MATCH_START_SCORE_NEAR_ZERO_THRESHOLD=20) を
    返し続けることで、score リセット境界の near_zero 条件を成立させる。
    """

    def __init__(self, score: int = 0) -> None:
        self._score = score

    def read_side(
        self, frame: np.ndarray, side: str,
    ) -> tuple[int | None, float]:
        return self._score, 1.0


def _make_pipe_with_trackers() -> RecognitionPipeline:
    """実 VideoChainTracker + スタブ score OCR 付き pipeline を構築する。"""
    from src.score_ocr import ScoreTracker

    stub_ocr = _StubScoreOcr(score=0)
    return RecognitionPipeline(
        image_reader=_StubImageReader(),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetector(),  # type: ignore[arg-type]
        score_ocr=stub_ocr,  # type: ignore[arg-type]
        chain_tracker_1p=VideoChainTracker(),
        chain_tracker_2p=VideoChainTracker(),
        stable_frame_count=2,
    )


def _dummy_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _board_2chain_540() -> Board:
    """2 連鎖 540 点の盤面 (tests/test_chain_detector.py と同構成)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[11][0] = COLOR_RED
    grid[12][0] = COLOR_RED
    grid[10][0] = COLOR_RED
    grid[9][0] = COLOR_RED
    grid[8][0] = COLOR_BLUE
    grid[7][0] = COLOR_BLUE
    grid[12][1] = COLOR_BLUE
    grid[11][1] = COLOR_BLUE
    grid[10][1] = COLOR_BLUE
    return Board.from_list(grid)


# ============================
# 1. reset(match_start_sec=...) の伝搬
# ============================


def test_reset_propagates_match_start_sec_to_trackers() -> None:
    """reset(match_start_sec=X) が再構築後の両 tracker に X を渡す。"""
    pipe = _make_pipe_with_trackers()
    pipe.reset(match_start_sec=500.0)
    assert pipe._chain_tracker_1p is not None
    assert pipe._chain_tracker_2p is not None
    assert pipe._chain_tracker_1p._match_start_sec == 500.0
    assert pipe._chain_tracker_2p._match_start_sec == 500.0


def test_reset_without_arg_is_backwards_compat_zero() -> None:
    """無引数 reset() は従来通り match_start_sec=0.0 (外部 caller 互換)。"""
    pipe = _make_pipe_with_trackers()
    pipe.reset()
    assert pipe._chain_tracker_1p is not None
    assert pipe._chain_tracker_1p._match_start_sec == 0.0
    assert pipe._chain_tracker_2p is not None
    assert pipe._chain_tracker_2p._match_start_sec == 0.0


def test_reset_none_trackers_no_error() -> None:
    """chain tracker 無し構成でも reset(match_start_sec=...) が壊れない。"""
    pipe = RecognitionPipeline(
        image_reader=_StubImageReader(),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetector(),  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
    )
    pipe.reset(match_start_sec=123.0)  # 例外が出ないこと
    assert pipe._chain_tracker_1p is None


# ============================
# 2. 境界検知 → reset(match_start_sec=time_sec) の配線 (W38 の本丸)
# ============================


def test_boundary_reset_passes_frame_time_as_match_start() -> None:
    """score リセット境界の内部 reset() が境界フレーム時刻を渡す。

    スタブ score OCR が両サイド 0 を返し続けるので、strict デバウンス
    (SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES=3) の 3 フレーム目で
    boundary_now が発火し reset(match_start_sec=time_sec) が呼ばれる。
    旧実装 (W38) では 0.0 のままだった。
    """
    pipe = _make_pipe_with_trackers()
    # 動画途中 t=400 秒から新試合が始まった、という状況
    times = [400.00, 400.05, 400.10, 400.15]
    for i, t in enumerate(times):
        pipe.update(i, t, _dummy_frame())
    assert pipe._chain_tracker_1p is not None
    assert pipe._chain_tracker_2p is not None
    # 境界発火フレーム (times のいずれか) の時刻が設定されている
    assert pipe._chain_tracker_1p._match_start_sec in times
    assert pipe._chain_tracker_1p._match_start_sec >= 400.0
    assert (
        pipe._chain_tracker_2p._match_start_sec
        == pipe._chain_tracker_1p._match_start_sec
    )


def test_chain_after_boundary_uses_match_relative_rate() -> None:
    """W38 根治の機能面: 境界後の連鎖はレート 70 で ojama_sent を計算する。

    2 連鎖 540 点を t=405 秒 (動画絶対) で発火:
        - 根治後: elapsed = 405 - 400.10 ≈ 4.9 秒 < 96 秒 → レート 70
          → ojama_sent = 540 // 70 = 7 (正しい値)
        - 旧実装: elapsed = 405 秒 > 320 秒 → レート 1
          → ojama_sent = 540 (点数がそのまま個数になる壊れ方)
    """
    pipe = _make_pipe_with_trackers()
    for i, t in enumerate([400.00, 400.05, 400.10, 400.15]):
        pipe.update(i, t, _dummy_frame())
    tracker = pipe._chain_tracker_1p
    assert tracker is not None
    tracker.update(405.0, _board_2chain_540())
    ev = tracker.update(405.5, Board())
    assert ev is not None
    assert ev.total_score == 540
    assert ev.ojama_sent == 7  # レート 70 (試合相対 ~5 秒)
    assert ev.leftover_score == 50


def test_chain_without_match_start_reproduces_w38() -> None:
    """旧挙動の記録: match_start_sec=0.0 のままだと t=405 でレート 1。

    (このテストは W38 が「なぜ壊れていたか」の実行可能な記録。
    VideoChainTracker 単体の既定値 0.0 の挙動自体は仕様通りで、
    壊れていたのは pipeline が試合開始時刻を渡していなかった配線。)
    """
    tracker = VideoChainTracker()  # match_start_sec 既定 0.0
    tracker.update(405.0, _board_2chain_540())
    ev = tracker.update(405.5, Board())
    assert ev is not None
    assert ev.total_score == 540
    # elapsed=405 秒 > 320 秒 → レート下限 1 → 点数がそのまま個数
    assert ev.ojama_sent == 540
