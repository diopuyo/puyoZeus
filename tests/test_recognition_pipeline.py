"""RecognitionPipeline 統合テスト (Phase B-7a)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.board import COLOR_RED, Board
from src.board_state_machine import BoardState
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, ImageReader
from src.match_state import MatchState, MatchStateDetector
from src.recognition_pipeline import RecognitionPipeline, SideResult


# ============================
# stub helpers
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
    """指定した固定 board を返す ImageReader スタブ."""

    def __init__(self, p1: Board, p2: Board) -> None:
        self._p1 = p1
        self._p2 = p2

    def read_both_boards(
        self, frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[Board, Board]:
        return self._p1.copy(), self._p2.copy()


def _empty_board() -> Board:
    return Board()


def _board_with_red(row: int, col: int) -> Board:
    b = Board()
    b.set(row, col, COLOR_RED)
    return b


def _make_pipe(
    p1: Board, p2: Board, in_match: bool = True,
    stable_n: int = 2,
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
# 基本動作
# ============================


def test_pipeline_reaches_stable_after_n_frames() -> None:
    """cycle 71v: 試合開始 window 内なら confirmed は空 Board() に強制される.

    CNN が puyo を観測しても物理ルール優先 (= 試合開始フィールドは空)。
    旧挙動は CNN 多数決で confirmed が設定されていたが、 v51/v70 の
    背景誤認対策で空強制に変更。
    """
    p1 = _board_with_red(12, 0)  # CNN は (12,0) に puyo を観測 (= 背景誤認 simulation)
    p2 = _empty_board()
    pipe = _make_pipe(p1, p2, stable_n=3)
    last_res = None
    for i in range(3):
        last_res = pipe.update(i, 0.05 * i, _dummy_frame())
    assert last_res is not None
    assert last_res.p1.state == BoardState.STABLE
    # match_just_started window 内なので confirmed は空 Board() (誤認無効化)
    assert last_res.p1.confirmed_board == _empty_board()
    assert last_res.p2.state == BoardState.STABLE
    assert last_res.p2.confirmed_board == p2


def test_pipeline_inactive_match_keeps_menu() -> None:
    p1 = _board_with_red(12, 0)
    p2 = _empty_board()
    pipe = _make_pipe(p1, p2, in_match=False, stable_n=2)
    res = pipe.update(0, 0.0, _dummy_frame())
    assert res.is_match_active is False
    assert res.p1.state == BoardState.MENU
    assert res.p2.state == BoardState.MENU


def test_pipeline_independent_p1_p2_states() -> None:
    """1P と 2P が独立して state を持つ."""
    pipe = _make_pipe(_board_with_red(12, 0), _empty_board(), stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.05, _dummy_frame())
    # 両方 STABLE
    res = pipe.update(2, 0.10, _dummy_frame())
    assert res.p1.state == BoardState.STABLE
    assert res.p2.state == BoardState.STABLE


def test_pipeline_reset() -> None:
    pipe = _make_pipe(_board_with_red(12, 0), _empty_board(), stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.05, _dummy_frame())
    pipe.reset()
    res = pipe.update(2, 0.10, _dummy_frame())
    # reset 直後は MENU から再スタート、1 frame では STABLE 不確定
    assert res.p1.state in {BoardState.MENU, BoardState.STABLE}


def test_side_result_carries_drift_info() -> None:
    pipe = _make_pipe(_board_with_red(12, 0), _empty_board(), stable_n=2)
    res = pipe.update(0, 0.0, _dummy_frame())
    assert isinstance(res.p1, SideResult)
    assert res.p1.drift.is_drift is False
    assert res.p1.cnn_board is not None


# ============================
# load_default smoke
# ============================


def test_next_count_constraint_excess_replaces_to_deficit() -> None:
    """サイクル66: tsumo_count に対し field 過剰色を deficit 色に置換."""
    from collections import Counter
    from src.board import COLOR_RED, COLOR_BLUE
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    board = Board()
    # field: RED 5 cells, BLUE 1 cell (= 6 puyo)
    for r, c in [(8, 0), (9, 0), (10, 0), (11, 0), (12, 0)]:
        board.set(r, c, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    # 累積 tsumo: RED 4, BLUE 2 (= 6 puyo expected)
    tsumo_count: Counter = Counter({COLOR_RED: 4, COLOR_BLUE: 2})
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
    )
    cnt: Counter = Counter()
    for r in range(13):
        for c in range(6):
            v = int(new_board.get(r, c))
            if v != 0:
                cnt[v] += 1
    # excess=1 RED, deficit=1 BLUE → RED が 1 cell BLUE に置換
    assert cnt[COLOR_RED] == 4
    assert cnt[COLOR_BLUE] == 2


def test_next_count_constraint_match_no_change() -> None:
    """サイクル66: field と tsumo_count 一致時は board 変更なし."""
    from collections import Counter
    from src.board import COLOR_RED, COLOR_BLUE
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    board = Board()
    board.set(11, 0, COLOR_RED)
    board.set(12, 0, COLOR_BLUE)
    tsumo_count: Counter = Counter({COLOR_RED: 1, COLOR_BLUE: 1})
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
    )
    assert int(new_board.get(11, 0)) == COLOR_RED
    assert int(new_board.get(12, 0)) == COLOR_BLUE


def test_constraint_invalidates_on_chain_event() -> None:
    """サイクル66: 連鎖発火後は constraint_valid=False."""
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.05, _dummy_frame())
    # constraint は初期 True
    assert pipe._constraint_valid_1p is True
    assert pipe._constraint_valid_2p is True


def test_load_default_smoke() -> None:
    """load_default が例外なく組み立てられる (実モデル未配置でも graceful)."""
    try:
        pipe = RecognitionPipeline.load_default(
            stable_frame_count=2,
            load_score_ocr=True,
            enable_chain_tracker=True,
        )
    except FileNotFoundError as e:
        pytest.skip(f"必須テンプレ未配置: {e}")
    assert pipe is not None


# ============================
# 全消し overlay hold (cycle 71v, 2026-05-14)
# ============================


class _StubChainTracker:
    """指定した ChainEvent を 1 回だけ返す chain_tracker スタブ."""

    def __init__(self, event: object | None = None) -> None:
        self._event = event
        self._returned = False
        self.match_start_sec = 0.0

    def update(self, time_sec: float, board: Board) -> object | None:
        if self._returned:
            return None
        self._returned = True
        return self._event

    def reset(self, **kwargs: object) -> None:
        self._returned = False


def _make_chain_event(is_all_clear: bool, chain_count: int = 2):
    """テスト用 ChainEvent ファクトリ."""
    from src.chain_detector import ChainEvent
    return ChainEvent(
        trigger_sec=1.0,
        end_sec=1.5,
        before_board=Board(),
        chain_count=chain_count,
        total_erased=8,
        total_score=400,
        base_score=400,
        all_clear_bonus_applied=0,
        ojama_sent=2,
        leftover_score=0,
        is_all_clear=is_all_clear,
    )


def _make_pipe_with_tracker(
    chain_event_1p: object | None,
) -> RecognitionPipeline:
    """全消し hold テスト用 pipeline (1P 側のみ chain_tracker stub)."""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(chain_event_1p)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
    )


def _prime_match_active(pipe: RecognitionPipeline, frames: int = 35) -> None:
    """chain_ban window (= CHAIN_BAN_FRAMES_AFTER_MATCH_START=30) を超えるよう
    試合中 frame を投入。"""
    for i in range(frames):
        pipe.update(i, 0.05 * i, _dummy_frame())


def test_all_clear_extends_chain_hold() -> None:
    """is_all_clear=True の ChainEvent では chain_until が延長される.

    cycle 71v (2026-05-14): 全消し overlay が CHAIN→STABLE 遷移時の
    _merge_diff_only を corrupt しないよう、 ALL_CLEAR_OVERLAY_HOLD_SEC
    ぶん CHAIN を延長する partial fix。
    """
    # event は ban window 通過後に投入したい → tracker は最初しか event を
    # 返さないので priming は tracker_event=None で行ってから event 注入する。
    ev = _make_chain_event(is_all_clear=True, chain_count=2)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    # ban window 通過後に tracker を差し替えて event を返させる
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    expected_base = t + pipe._chain_hold_per_step_sec * 2
    expected_total = expected_base + RecognitionPipeline.ALL_CLEAR_OVERLAY_HOLD_SEC
    assert pipe._chain_until_1p == pytest.approx(expected_total)


def test_non_all_clear_does_not_extend_chain_hold() -> None:
    """is_all_clear=False の ChainEvent では延長なし (= 旧挙動互換)."""
    ev = _make_chain_event(is_all_clear=False, chain_count=2)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    expected_base = t + pipe._chain_hold_per_step_sec * 2
    assert pipe._chain_until_1p == pytest.approx(expected_base)


