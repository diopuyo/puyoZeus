"""NextSlideDetector + tsumo placement validator のテスト (Phase I R-1/R-7)."""

from __future__ import annotations

import numpy as np
import pytest

from src.board import COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, Board
from src.board_state_machine import (
    BoardState,
    DetectorSignals,
    StateContext,
)
from src.next_detector import (
    ROI_1P_DNEXT_BOT,
    ROI_1P_DNEXT_TOP,
    ROI_1P_NEXT_BOT,
    ROI_1P_NEXT_TOP,
)
from src.next_slide_detector import (
    DEFAULT_DIFF_THRESHOLD,
    NextSlideDetector,
    PlacementValidationResult,
    SlideMotionResult,
    validate_tsumo_placement,
)
from src.state_detectors import TsumoPhaseDetector


# ============================
# helper
# ============================


def _blank_frame() -> np.ndarray:
    """全 0 BGR uint8 の 1080x1920 frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _frame_with_next_color(rgb_value: int) -> np.ndarray:
    """1P next/dnext ROI 領域に rgb_value で puyo 風に塗った frame."""
    f = _blank_frame()
    for roi in (
        ROI_1P_NEXT_TOP, ROI_1P_NEXT_BOT,
        ROI_1P_DNEXT_TOP, ROI_1P_DNEXT_BOT,
    ):
        y1, y2, x1, x2 = roi
        f[y1:y2, x1:x2] = rgb_value
    return f


def _board_with(positions: list[tuple[int, int, int]]) -> Board:
    b = Board()
    for r, c, color in positions:
        b.set(r, c, color)
    return b


# ============================
# NextSlideDetector unit tests
# ============================


def test_slide_detector_invalid_side_raises() -> None:
    with pytest.raises(ValueError):
        NextSlideDetector(side="3P")  # type: ignore[arg-type]


def test_slide_detector_returns_no_motion_for_none_inputs() -> None:
    det = NextSlideDetector(side="1P")
    res = det.update(None, None)
    assert res.slide_motion is False
    assert res.diff_score == 0.0


def test_slide_detector_no_motion_for_identical_frames() -> None:
    det = NextSlideDetector(side="1P")
    f = _frame_with_next_color(100)
    res = det.update(f, f)
    assert res.slide_motion is False
    # 完全一致なら diff_score = 0.0
    assert res.diff_score == pytest.approx(0.0)


def test_slide_detector_fires_when_next_roi_changes_drastically() -> None:
    det = NextSlideDetector(side="1P")
    prev = _frame_with_next_color(50)
    curr = _frame_with_next_color(200)  # 150 程度の差分
    res = det.update(prev, curr)
    assert res.slide_motion is True
    assert res.diff_score > DEFAULT_DIFF_THRESHOLD


def test_slide_detector_cooldown_prevents_repeat_fire() -> None:
    det = NextSlideDetector(side="1P", cooldown_frames=3)
    prev = _frame_with_next_color(50)
    curr = _frame_with_next_color(200)
    r1 = det.update(prev, curr)
    assert r1.slide_motion is True
    # cooldown 中の連続発火は抑制
    r2 = det.update(prev, curr)
    assert r2.slide_motion is False


def test_slide_detector_reset_clears_state() -> None:
    det = NextSlideDetector(side="1P")
    prev = _frame_with_next_color(50)
    curr = _frame_with_next_color(200)
    det.update(prev, curr)
    det.reset()
    # cooldown が消えて再発火可能
    r = det.update(prev, curr)
    assert r.slide_motion is True


def test_slide_detector_2p_uses_2p_rois() -> None:
    """2P side instance は 1P ROI を見ない (= 1P 領域のみ変えても発火しない)."""
    det = NextSlideDetector(side="2P")
    prev = _blank_frame()
    curr = _frame_with_next_color(200)  # 1P 領域のみ変化
    res = det.update(prev, curr)
    assert res.slide_motion is False


# ============================
# validate_tsumo_placement unit tests
# ============================


def test_validate_returns_inconsistent_for_none_baseline() -> None:
    landed = _board_with([(12, 0, COLOR_RED)])
    res = validate_tsumo_placement(None, landed, (COLOR_RED, COLOR_BLUE))
    assert isinstance(res, PlacementValidationResult)
    assert res.consistent is False


def test_validate_returns_inconsistent_for_none_falling_pair() -> None:
    base = Board()
    landed = _board_with([(12, 0, COLOR_RED)])
    res = validate_tsumo_placement(base, landed, None)
    assert res.consistent is False


def test_validate_returns_consistent_for_proper_two_color_drop() -> None:
    """空盤面に red + blue のツモが置かれた → consistent=True."""
    base = Board()
    landed = _board_with([(12, 0, COLOR_RED), (11, 0, COLOR_BLUE)])
    res = validate_tsumo_placement(base, landed, (COLOR_BLUE, COLOR_RED))
    assert res.consistent is True
    assert res.delta_total == 2
    assert res.details[COLOR_RED] == 1
    assert res.details[COLOR_BLUE] == 1


def test_validate_returns_inconsistent_when_color_differs() -> None:
    """落下ペアに無い色が増えていれば inconsistent."""
    base = Board()
    landed = _board_with([(12, 0, COLOR_GREEN), (11, 0, COLOR_YELLOW)])
    res = validate_tsumo_placement(base, landed, (COLOR_RED, COLOR_BLUE))
    assert res.consistent is False


def test_validate_tolerates_one_off_for_chain_edge_case() -> None:
    """連鎖発火寸前等で 1 個ずれても tolerance で許容."""
    base = Board()
    # 期待 +2 だが実測 +1 (1 個消えた等)、tolerance=1 なら consistent
    landed = _board_with([(12, 0, COLOR_RED)])
    res = validate_tsumo_placement(
        base, landed, (COLOR_RED, COLOR_BLUE), tolerance=1,
    )
    # red +1 で blue 期待 +1 だが actual=0、|0-1|=1 で許容範囲内
    # delta_total=1 で 1 in [1-1, 2+1] → consistent
    assert res.consistent is True


def test_validate_returns_inconsistent_for_excessive_increase() -> None:
    base = Board()
    # 5 個増 = 連鎖中などの異常
    landed = _board_with([
        (12, 0, COLOR_RED), (11, 0, COLOR_BLUE),
        (12, 1, COLOR_RED), (11, 1, COLOR_BLUE),
        (12, 2, COLOR_RED),
    ])
    res = validate_tsumo_placement(base, landed, (COLOR_RED, COLOR_BLUE))
    assert res.consistent is False


# ============================
# Tsumo + slide_motion 統合テスト
# ============================


def _signal(
    t: float, board: Board, *,
    chain_event: object | None = None,
    score_delta: int = 0,
    next_pair: tuple[int, int] | None = None,
    slide_motion: bool = False,
    placement_validated: bool = False,
    match: bool = True,
) -> DetectorSignals:
    return DetectorSignals(
        time_sec=t,
        cnn_board=board,
        is_match_active=match,
        chain_event=chain_event,
        score_delta=score_delta,
        next_pair=next_pair,
        slide_motion=slide_motion,
        placement_validated=placement_validated,
    )


def test_tsumo_detector_slide_motion_forces_stable_immediately() -> None:
    """TSUMO_FALL 中に slide_motion=True なら landed_consec を待たず STABLE."""
    det = TsumoPhaseDetector(consec_threshold=2, landed_consec=10)
    base = Board()
    # baseline 確定 + TSUMO_FALL state 突入
    ctx = StateContext(state=BoardState.TSUMO_FALL, confirmed_board=base)
    landed_board = _board_with([(12, 0, COLOR_RED), (11, 0, COLOR_BLUE)])
    # 通常なら landed_consec=10 だが、slide_motion=True なら即時 STABLE
    res = det.detect(
        ctx, _signal(1.0, landed_board, slide_motion=True),
    )
    assert res == BoardState.STABLE


def test_tsumo_detector_slide_motion_outside_tsumo_state_ignored() -> None:
    """STABLE state 中の slide_motion は遷移を起こさない (CHAIN/STABLE 担当外)."""
    det = TsumoPhaseDetector()
    base = Board()
    ctx = StateContext(state=BoardState.STABLE, confirmed_board=base)
    res = det.detect(ctx, _signal(1.0, base, slide_motion=True))
    # diff=0 で STABLE 維持 (None)
    assert res is None


def test_tsumo_detector_placement_validated_speeds_up_landing() -> None:
    """placement_validated=True なら landed_consec が 1 緩和される."""
    det = TsumoPhaseDetector(consec_threshold=2, landed_consec=3)
    base = Board()
    ctx = StateContext(state=BoardState.TSUMO_FALL, confirmed_board=base)
    landed_board = _board_with([(12, 0, COLOR_RED), (11, 0, COLOR_BLUE)])
    # 1 frame 目: same=False で landed_consec_count=1 (= 3-1=2 未満)
    r1 = det.detect(
        ctx, _signal(1.0, landed_board, placement_validated=True),
    )
    assert r1 is None
    # 2 frame 目: same=True で 2 達成 → STABLE
    r2 = det.detect(
        ctx, _signal(1.1, landed_board, placement_validated=True),
    )
    assert r2 == BoardState.STABLE


def test_tsumo_detector_no_slide_no_validation_uses_default_landed() -> None:
    """slide=False, validated=False なら従来通り landed_consec frame 待つ."""
    det = TsumoPhaseDetector(consec_threshold=2, landed_consec=2)
    base = Board()
    ctx = StateContext(state=BoardState.TSUMO_FALL, confirmed_board=base)
    landed_board = _board_with([(12, 0, COLOR_RED), (11, 0, COLOR_BLUE)])
    r1 = det.detect(ctx, _signal(1.0, landed_board))
    # 1 回目: count=1 で未達 → None
    assert r1 is None
    r2 = det.detect(ctx, _signal(1.1, landed_board))
    # 2 回目: count=2 で達成 → STABLE
    assert r2 == BoardState.STABLE


def test_tsumo_detector_slide_motion_with_chain_diff_skipped() -> None:
    """puyo count が +max_increase 超なら slide_motion 無視 (CHAIN 任せ)."""
    det = TsumoPhaseDetector(consec_threshold=2, landed_consec=2)
    base = Board()
    ctx = StateContext(state=BoardState.TSUMO_FALL, confirmed_board=base)
    big_board = _board_with([
        (12, 0, COLOR_RED), (11, 0, COLOR_BLUE),
        (12, 1, COLOR_RED),
    ])  # +3 個
    res = det.detect(ctx, _signal(1.0, big_board, slide_motion=True))
    # +3 は max_increase=2 を超えるので CHAIN detector 任せ → None
    # (= TSUMO_FALL を維持、landed_consec 経路のみ)
    # ただし same=False で landed_consec_count=1、未達で None
    assert res is None
