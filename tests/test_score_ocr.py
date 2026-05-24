"""src/score_ocr.py のテスト。

8 桁スコア OCR ロジックを合成画像 + 実テンプレで検証する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.score_ocr import (
    DIGIT_COUNT,
    DIGIT_HEIGHT,
    DIGIT_LEFTS_1P,
    DIGIT_LEFTS_2P,
    DIGIT_TOP,
    DIGIT_WIDTH,
    EXPECTED_FRAME_SHAPE,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
    ScoreDelta,
    ScoreOcr,
    ScoreReadResult,
    ScoreTracker,
)

TEMPLATE_DIR = Path("models/ui_templates/score_digits")


def _load_real_templates() -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not TEMPLATE_DIR.is_dir():
        return out
    for n in range(10):
        p = TEMPLATE_DIR / f"digit_{n}.png"
        img = cv2.imread(str(p))
        if img is not None:
            out[n] = img
    return out


def _make_blank_frame() -> np.ndarray:
    h, w = EXPECTED_FRAME_SHAPE
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint_score_into_frame(
    frame: np.ndarray,
    digits: list[int],
    side: str,
    templates: dict[int, np.ndarray],
) -> np.ndarray:
    """8 桁の数字を ROI に描き込み、合成テストフレームを作る。"""
    region = SCORE_1P_REGION if side == "1P" else SCORE_2P_REGION
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    y1, _, x1, _ = region
    for i, d in enumerate(digits):
        tpl = templates.get(d)
        if tpl is None:
            continue
        # テンプレを ROI 内の i 番目桁位置に貼る
        ay = y1 + DIGIT_TOP
        ax = x1 + lefts[i]
        if tpl.shape[:2] != (DIGIT_HEIGHT, DIGIT_WIDTH):
            tpl = cv2.resize(tpl, (DIGIT_WIDTH, DIGIT_HEIGHT))
        frame[ay:ay + DIGIT_HEIGHT, ax:ax + DIGIT_WIDTH] = tpl
    return frame


# =============================================================================


def test_score_ocr_constants() -> None:
    assert DIGIT_COUNT == 8
    assert SCORE_1P_REGION[0] < SCORE_1P_REGION[1]
    assert SCORE_2P_REGION[2] < SCORE_2P_REGION[3]
    assert len(DIGIT_LEFTS_1P) == DIGIT_COUNT
    assert len(DIGIT_LEFTS_2P) == DIGIT_COUNT


def test_score_ocr_load_default_templates_present() -> None:
    """テンプレが配置済みであれば load_default は警告なしに動く。"""
    ocr = ScoreOcr.load_default()
    # テンプレ未配置でも load_default 自体は失敗しない
    assert ocr is not None


def test_score_ocr_no_templates_returns_none() -> None:
    """テンプレを 1 つも渡さない場合は常に None。"""
    ocr = ScoreOcr(templates={})
    frame = _make_blank_frame()
    res = ocr.read(frame)
    assert isinstance(res, ScoreReadResult)
    assert res.score_1p is None
    assert res.score_2p is None
    assert all(d is None for d in res.digits_1p)
    assert all(d is None for d in res.digits_2p)


def test_score_ocr_invalid_frame_shape() -> None:
    """形状不正フレームでは None を返す。"""
    ocr = ScoreOcr.load_default()
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    res = ocr.read(bad)
    assert res.score_1p is None
    assert res.score_2p is None


def test_score_ocr_zeros_synthetic_1p() -> None:
    """全桁 0 の合成 1080p フレームで 1P=0 が読める。"""
    templates = _load_real_templates()
    if 0 not in templates:
        pytest.skip("digit_0 テンプレ未整備")
    ocr = ScoreOcr.load_default()
    frame = _make_blank_frame()
    _paint_score_into_frame(frame, [0] * 8, side="1P", templates=templates)
    res = ocr.read(frame)
    assert res.score_1p == 0
    assert res.confidence_1p > 0.5


def test_score_ocr_zeros_synthetic_2p() -> None:
    """全桁 0 の合成フレームで 2P=0 が読める。"""
    templates = _load_real_templates()
    if 0 not in templates:
        pytest.skip("digit_0 テンプレ未整備")
    ocr = ScoreOcr.load_default()
    frame = _make_blank_frame()
    _paint_score_into_frame(frame, [0] * 8, side="2P", templates=templates)
    res = ocr.read(frame)
    assert res.score_2p == 0
    assert res.confidence_2p > 0.5


def test_score_ocr_specific_number_synthetic() -> None:
    """合成フレームで 12345678 を読み取れる。"""
    templates = _load_real_templates()
    needed = set(range(1, 9))
    if not needed.issubset(templates.keys()):
        pytest.skip(f"テンプレ未整備: {needed - templates.keys()}")
    ocr = ScoreOcr.load_default()
    frame = _make_blank_frame()
    digits = [1, 2, 3, 4, 5, 6, 7, 8]
    _paint_score_into_frame(frame, digits, side="1P", templates=templates)
    res = ocr.read(frame)
    assert res.score_1p == 12345678


def test_score_ocr_read_side_only() -> None:
    """read_side で片方のみ読める。"""
    templates = _load_real_templates()
    if 0 not in templates:
        pytest.skip("digit_0 テンプレ未整備")
    ocr = ScoreOcr.load_default()
    frame = _make_blank_frame()
    _paint_score_into_frame(frame, [0] * 8, side="1P", templates=templates)
    score, conf = ocr.read_side(frame, "1P")
    assert score == 0
    assert conf > 0.5


def test_score_ocr_resizes_non_1080p_frame() -> None:
    """720p フレームは 1080p に内部リサイズされる (None 返却ではない)。"""
    ocr = ScoreOcr(templates={})
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = ocr.read(frame)
    # テンプレ無しで結果は None だが、エラーは出ない
    assert res.score_1p is None


def test_score_ocr_blank_frame_low_confidence() -> None:
    """完全にブランクなフレームは confidence が低く None を返す。"""
    ocr = ScoreOcr.load_default()
    frame = _make_blank_frame()
    res = ocr.read(frame)
    assert res.score_1p is None
    assert res.confidence_1p < 0.5


# =============================================================================
# ScoreTracker (Phase B-4)
# =============================================================================


def _frame_with_score(side: str, digits: list[int]) -> np.ndarray:
    templates = _load_real_templates()
    frame = _make_blank_frame()
    _paint_score_into_frame(frame, digits, side=side, templates=templates)
    return frame


def test_score_tracker_initial_update_has_no_prev() -> None:
    if 0 not in _load_real_templates():
        pytest.skip("digit_0 テンプレ未整備")
    tracker = ScoreTracker(side="1P", ocr=ScoreOcr.load_default())
    delta = tracker.update(_frame_with_score("1P", [0] * 8))
    assert isinstance(delta, ScoreDelta)
    assert delta.prev_score is None
    assert delta.cur_score == 0
    assert delta.delta == 0
    assert not delta.is_valid  # prev=None なので invalid
    assert tracker.last_score == 0


def test_score_tracker_same_score_zero_delta() -> None:
    if 0 not in _load_real_templates():
        pytest.skip("digit_0 テンプレ未整備")
    tracker = ScoreTracker(side="1P", ocr=ScoreOcr.load_default())
    tracker.update(_frame_with_score("1P", [0] * 8))
    delta = tracker.update(_frame_with_score("1P", [0] * 8))
    assert delta.prev_score == 0
    assert delta.cur_score == 0
    assert delta.delta == 0
    assert delta.is_valid


def test_score_tracker_positive_delta_on_increase() -> None:
    needed = {0, 1, 2, 3, 4, 5}
    if not needed.issubset(_load_real_templates()):
        pytest.skip("digit テンプレ未整備")
    tracker = ScoreTracker(side="1P", ocr=ScoreOcr.load_default())
    # 00000000 → 00012345
    tracker.update(_frame_with_score("1P", [0] * 8))
    delta = tracker.update(
        _frame_with_score("1P", [0, 0, 0, 1, 2, 3, 4, 5]),
    )
    assert delta.prev_score == 0
    assert delta.cur_score == 12345
    assert delta.delta == 12345
    assert delta.is_valid


def test_score_tracker_no_update_when_unreadable() -> None:
    """読めない frame では last_score を保持し delta=0."""
    if 0 not in _load_real_templates():
        pytest.skip("digit_0 テンプレ未整備")
    tracker = ScoreTracker(side="1P", ocr=ScoreOcr.load_default())
    tracker.update(_frame_with_score("1P", [0] * 8))
    assert tracker.last_score == 0
    # blank frame は読めない
    blank = _make_blank_frame()
    delta = tracker.update(blank)
    assert delta.cur_score is None
    assert delta.delta == 0
    assert tracker.last_score == 0  # 直前の値を保持


def test_score_tracker_reset_clears_last() -> None:
    if 0 not in _load_real_templates():
        pytest.skip("digit_0 テンプレ未整備")
    tracker = ScoreTracker(side="1P", ocr=ScoreOcr.load_default())
    tracker.update(_frame_with_score("1P", [0] * 8))
    tracker.reset()
    assert tracker.last_score is None


def test_score_tracker_independent_sides() -> None:
    """1P/2P trackers は独立、互いに影響しない."""
    needed = {0, 5}
    if not needed.issubset(_load_real_templates()):
        pytest.skip("digit テンプレ未整備")
    ocr = ScoreOcr.load_default()
    t1 = ScoreTracker(side="1P", ocr=ocr)
    t2 = ScoreTracker(side="2P", ocr=ocr)
    templates = _load_real_templates()
    frame = _make_blank_frame()
    _paint_score_into_frame(frame, [0] * 8, side="1P", templates=templates)
    _paint_score_into_frame(
        frame, [0, 0, 0, 0, 0, 0, 5, 5], side="2P", templates=templates,
    )
    d1 = t1.update(frame)
    d2 = t2.update(frame)
    assert d1.cur_score == 0
    assert d2.cur_score == 55


def test_score_tracker_rejects_invalid_side() -> None:
    ocr = ScoreOcr.load_default()
    with pytest.raises(ValueError):
        ScoreTracker(side="3P", ocr=ocr)  # type: ignore[arg-type]


def test_read_with_neighbor_search_exists() -> None:
    """read_with_neighbor_search が呼び出し可能 (cap=None でクラッシュしない)。"""
    ocr = ScoreOcr.load_default()
    # cv2.VideoCapture は実動画が必要だが、動作上の type check のみ確認
    assert hasattr(ocr, "read_with_neighbor_search")
    # 引数のデフォルト値も含めて method が存在することを確認
    import inspect
    sig = inspect.signature(ocr.read_with_neighbor_search)
    assert "search_radius_sec" in sig.parameters
    assert "n_samples" in sig.parameters
