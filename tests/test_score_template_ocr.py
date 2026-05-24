"""src/score_template_ocr.py のテスト.

pixel-level digit テンプレートマッチングによる score OCR 経路を検証する。
合成 ROI (テンプレ画像を所定位置に貼り付け) で正確に元の score を復元できる
ことを主に確認する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.score_ocr import (
    DIGIT_HEIGHT,
    DIGIT_LEFTS_1P,
    DIGIT_LEFTS_2P,
    DIGIT_TOP,
    DIGIT_WIDTH,
)
from src.score_template_ocr import (
    DEFAULT_FALLBACK_THRESHOLD,
    classify_digit_cell,
    load_default_templates,
    make_score_ocr,
    recognize_digits,
    segment_digits,
)

TEMPLATE_DIR = Path("models/ui_templates/score_digits")
ROI_HEIGHT = 65
ROI_WIDTH = 320


# ======================================================================
# helpers
# ======================================================================


def _load_templates() -> dict[int, np.ndarray]:
    """実テンプレを読込 (なければ空 dict)。"""
    out: dict[int, np.ndarray] = {}
    if not TEMPLATE_DIR.is_dir():
        return out
    for n in range(10):
        p = TEMPLATE_DIR / f"digit_{n}.png"
        img = cv2.imread(str(p))
        if img is not None:
            out[n] = img
    return out


def _make_roi(
    digits: list[int],
    templates: dict[int, np.ndarray],
    side: str = "1P",
) -> np.ndarray:
    """digit 列と テンプレから 65x320 BGR ROI を合成."""
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    for i, d in enumerate(digits):
        tpl = templates.get(d)
        if tpl is None:
            continue
        if tpl.shape[:2] != (DIGIT_HEIGHT, DIGIT_WIDTH):
            tpl = cv2.resize(tpl, (DIGIT_WIDTH, DIGIT_HEIGHT))
        x = lefts[i]
        roi[
            DIGIT_TOP:DIGIT_TOP + DIGIT_HEIGHT,
            x:x + DIGIT_WIDTH,
        ] = tpl
    return roi


# ======================================================================
# load_default_templates
# ======================================================================


def test_load_default_templates_keys_subset() -> None:
    """読込結果の key は 0..9 の subset であること。"""
    tpl = load_default_templates()
    assert all(0 <= k <= 9 for k in tpl.keys())


def test_load_default_templates_missing_dir() -> None:
    """存在しないディレクトリでも空 dict を返す。"""
    tpl = load_default_templates(template_dir=Path("nonexistent_dir_xyz"))
    assert tpl == {}


# ======================================================================
# segment_digits
# ======================================================================


def test_segment_digits_default_count() -> None:
    """default は 8 桁、各矩形の幅高さは固定値."""
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    segs = segment_digits(roi)
    assert len(segs) == 8
    for (x, y, w, h) in segs:
        assert w == DIGIT_WIDTH
        assert h == DIGIT_HEIGHT
        assert y == DIGIT_TOP


def test_segment_digits_x_positions_match_constants() -> None:
    """x 座標は DIGIT_LEFTS_1P と一致."""
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    segs = segment_digits(roi, side="1P")
    xs = [s[0] for s in segs]
    assert tuple(xs) == DIGIT_LEFTS_1P


def test_segment_digits_invalid_count_raises() -> None:
    """digit_count が範囲外なら ValueError."""
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        segment_digits(roi, digit_count=99)


def test_segment_digits_zero_count_returns_empty() -> None:
    """digit_count=0 で空リスト."""
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    assert segment_digits(roi, digit_count=0) == []


# ======================================================================
# classify_digit_cell
# ======================================================================


def test_classify_digit_cell_empty_templates_returns_none() -> None:
    """テンプレ空なら必ず None."""
    cell = np.zeros((DIGIT_HEIGHT, DIGIT_WIDTH, 3), dtype=np.uint8)
    label, conf = classify_digit_cell(cell, {})
    assert label is None
    assert conf == 0.0


def test_classify_digit_cell_recovers_each_digit() -> None:
    """各 digit テンプレを直接渡したら自身を高 confidence で classify."""
    tpl = _load_templates()
    if len(tpl) < 10:
        pytest.skip("テンプレ 10 個揃っていない")
    # classify_digit_cell は内部で resize+gray 化する → BGR テンプレを直接渡す
    from src.score_template_ocr import _normalize_template_set
    tpl_gray = _normalize_template_set(tpl)
    for d, img in tpl.items():
        label, conf = classify_digit_cell(img, tpl_gray)
        assert label == d, f"digit {d} を {label} と誤判定 (conf={conf:.3f})"
        assert conf > 0.9


# ======================================================================
# recognize_digits — 主要 API
# ======================================================================


def test_recognize_digits_zeros() -> None:
    """全桁 0 の合成 ROI から score=0 を復元."""
    tpl = _load_templates()
    if 0 not in tpl:
        pytest.skip("digit_0 テンプレ未整備")
    roi = _make_roi([0] * 8, tpl)
    score, confs = recognize_digits(roi, tpl)
    assert score == 0
    assert len(confs) == 8
    assert all(c > 0.7 for c in confs)


def test_recognize_digits_each_position() -> None:
    """0..9 を含む 8 桁列を正確に復元 (タスク要求)."""
    tpl = _load_templates()
    if len(tpl) < 10:
        pytest.skip("テンプレ 10 個揃っていない")
    digits = [1, 2, 3, 4, 5, 6, 7, 9]  # 0..9 を満遍なく
    roi = _make_roi(digits, tpl)
    score, confs = recognize_digits(roi, tpl)
    assert score == 12345679
    assert all(c > 0.7 for c in confs)


def test_recognize_digits_with_zero_at_front() -> None:
    """先頭ゼロ込みの 00012340 を読み取り."""
    tpl = _load_templates()
    if len(tpl) < 5:
        pytest.skip("テンプレ不足")
    digits = [0, 0, 0, 1, 2, 3, 4, 0]
    roi = _make_roi(digits, tpl)
    score, _ = recognize_digits(roi, tpl)
    assert score == 12340


def test_recognize_digits_2p_side() -> None:
    """side='2P' でも同じ結果 (DIGIT_LEFTS_2P と 1P が同値だが API 確認)."""
    tpl = _load_templates()
    if 0 not in tpl:
        pytest.skip("テンプレ未整備")
    digits = [9, 8, 7, 6, 5, 4, 3, 2]
    if not all(d in tpl for d in digits):
        pytest.skip("テンプレ不足")
    roi = _make_roi(digits, tpl, side="2P")
    score, _ = recognize_digits(roi, tpl, side="2P")
    assert score == 98765432


def test_recognize_digits_blank_roi_returns_none() -> None:
    """ブランク ROI → None + 全 conf 低い."""
    tpl = _load_templates()
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    score, confs = recognize_digits(roi, tpl)
    assert score is None
    assert len(confs) == 8


def test_recognize_digits_empty_input_returns_none() -> None:
    """size=0 入力 → None."""
    tpl = _load_templates()
    score, confs = recognize_digits(np.zeros((0, 0, 3), dtype=np.uint8), tpl)
    assert score is None
    assert confs == [0.0] * 8


def test_recognize_digits_no_templates_returns_none() -> None:
    """テンプレ空 → None + 全 conf 0."""
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    score, confs = recognize_digits(roi, {})
    assert score is None
    assert confs == [0.0] * 8


def test_recognize_digits_per_digit_confidence_range() -> None:
    """成功時 per_digit_confidence は [0, 1] に収まる."""
    tpl = _load_templates()
    if 0 not in tpl:
        pytest.skip("テンプレ未整備")
    roi = _make_roi([0] * 8, tpl)
    _, confs = recognize_digits(roi, tpl)
    assert all(0.0 <= c <= 1.0 for c in confs)


# ======================================================================
# fallback hook
# ======================================================================


def test_recognize_digits_fallback_invoked_on_low_confidence() -> None:
    """ブランク ROI → confidence 不足 → fallback_fn が呼ばれる."""
    tpl = _load_templates()
    if not tpl:
        pytest.skip("テンプレ未整備")
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    called: list[bool] = []

    def fb(_roi, _segs, _confs):
        called.append(True)
        return (12345678, [0.99] * 8)

    score, confs = recognize_digits(roi, tpl, fallback_fn=fb)
    assert called == [True]
    assert score == 12345678
    assert confs == [0.99] * 8


def test_recognize_digits_fallback_returning_none_propagates() -> None:
    """fallback_fn が None を返すと最終結果も None."""
    tpl = _load_templates()
    if not tpl:
        pytest.skip("テンプレ未整備")
    roi = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    score, _ = recognize_digits(
        roi, tpl, fallback_fn=lambda *_args: None,
    )
    assert score is None


def test_recognize_digits_fallback_not_invoked_on_success() -> None:
    """成功時は fallback_fn を呼ばない."""
    tpl = _load_templates()
    if 0 not in tpl:
        pytest.skip("テンプレ未整備")
    roi = _make_roi([0] * 8, tpl)
    called: list[bool] = []

    def fb(_roi, _segs, _confs):
        called.append(True)
        return (-1, [0.0] * 8)

    score, _ = recognize_digits(roi, tpl, fallback_fn=fb)
    assert called == []
    assert score == 0


def test_recognize_digits_fallback_threshold_trigger() -> None:
    """fallback_threshold を意図的に高くすると fallback が起動."""
    tpl = _load_templates()
    if 0 not in tpl:
        pytest.skip("テンプレ未整備")
    roi = _make_roi([0] * 8, tpl)
    called: list[bool] = []

    def fb(_roi, _segs, _confs):
        called.append(True)
        return None

    # 1.5 は到達不能な閾値 → 必ず fallback 起動
    score, _ = recognize_digits(
        roi, tpl,
        fallback_threshold=1.5,
        fallback_fn=fb,
    )
    assert called == [True]
    # fallback が None を返したので最終 None
    assert score is None


# ======================================================================
# default fallback threshold
# ======================================================================


def test_default_fallback_threshold_is_reasonable() -> None:
    """0.5 ~ 0.9 の妥当な範囲."""
    assert 0.5 <= DEFAULT_FALLBACK_THRESHOLD <= 0.9


# ======================================================================
# make_score_ocr (既存 ScoreOcr ブリッジ)
# ======================================================================


def test_make_score_ocr_default() -> None:
    """make_score_ocr は ScoreOcr インスタンスを返す."""
    from src.score_ocr import ScoreOcr
    ocr = make_score_ocr()
    assert isinstance(ocr, ScoreOcr)


def test_make_score_ocr_use_template_false_raises() -> None:
    """use_template_ocr=False は将来実装の予約 → NotImplementedError."""
    with pytest.raises(NotImplementedError):
        make_score_ocr(use_template_ocr=False)
