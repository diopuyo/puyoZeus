"""src/chain_count_truth.py のテスト。

2系統 (テロップ読み/得点逆算) の突合ロジックを、
- 実データ (video_c54、chain_count_ocr.py docstring/tests に記録済みの実測値)
- 合成データ (境界条件・fail-safe 経路)
の両方で検証する。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.chain_count_ocr import (
    CHAIN_DIGIT_LABELS,
    ChainCountOcr,
    ChainCountReadResult,
    ChainCountWindowResult,
    EXPECTED_FRAME_SHAPE,
    _approx_min_chain_score,
)
from src.chain_count_truth import (
    FULL_CHAIN_COUNT_CANDIDATES,
    HIGH_CONFIDENCE_SCORE_RATIO_MAX,
    HIGH_CONFIDENCE_SCORE_RATIO_MIN,
    TELOP_SEARCH_POST_BUFFER_SEC,
    TELOP_SEARCH_PRE_BUFFER_SEC,
    ChainCountTruthResult,
    HighConfidenceScoreResult,
    compute_telop_search_window,
    read_chain_count_truth,
    resolve_chain_count_truth,
    select_chain_count_high_confidence_band,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

TEMPLATE_DIR = Path("models/ui_templates/chain_count_digits")


def _load_real_templates() -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not TEMPLATE_DIR.is_dir():
        return out
    for n in CHAIN_DIGIT_LABELS:
        p = TEMPLATE_DIR / f"digit_{n}.png"
        img = cv2.imread(str(p))
        if img is not None:
            out[n] = img
    return out


def _make_blank_frame() -> np.ndarray:
    h, w = EXPECTED_FRAME_SHAPE
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _paint_digit_into_board(
    frame: np.ndarray, digit_img: np.ndarray, side: str,
    offset_x: int, offset_y: int,
) -> np.ndarray:
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    ax = region.x + offset_x
    ay = region.y + offset_y
    h, w = digit_img.shape[:2]
    frame[ay:ay + h, ax:ax + w] = digit_img
    return frame


class _FakeVideoCapture:
    """cv2.VideoCapture の set()/read() のみを模したフェイク (test_chain_count_ocr.py と同型)。"""

    def __init__(self, frame_by_time_sec: dict[float, np.ndarray], default: np.ndarray) -> None:
        self._frame_by_time = frame_by_time_sec
        self._default = default
        self._cur_time_ms: float = 0.0

    def set(self, prop: int, value: float) -> None:
        self._cur_time_ms = value

    def read(self) -> tuple[bool, np.ndarray]:
        t_sec = round(self._cur_time_ms / 1000.0, 2)
        if not self._frame_by_time:
            return True, self._default
        closest = min(self._frame_by_time, key=lambda k: abs(k - t_sec))
        return True, self._frame_by_time[closest]


# =============================================================================
# 定数
# =============================================================================


def test_full_candidates_covers_chain_count_range() -> None:
    assert min(FULL_CHAIN_COUNT_CANDIDATES) == 1
    assert max(FULL_CHAIN_COUNT_CANDIDATES) == 19
    assert len(FULL_CHAIN_COUNT_CANDIDATES) == 19


# =============================================================================
# resolve_chain_count_truth: 実データ検証 (video_c54)
# =============================================================================


def test_resolve_agrees_on_real_video_c54_1p_5chain() -> None:
    """video_c54 1P game_idx=1 (実測 delta_score=7598、真の連鎖数5)。

    テロップが正しく5を読めていれば (連続列方式が壊れていないケース)、
    得点逆算 (5) と一致し真値5を採用できる。
    """
    telop = ChainCountWindowResult(
        max_chain_count=5,
        samples=(ChainCountReadResult(5, 0.9, (0, 0)),),
        n_hits=1,
        method="monotonic_run",
    )
    result = resolve_chain_count_truth(telop, delta_score=7598)
    assert isinstance(result, ChainCountTruthResult)
    assert result.chain_count == 5
    assert result.reason == "agree"
    assert result.score_chain_count == 5


def test_resolve_disagrees_on_real_video_c54_2p_broken_telop_run() -> None:
    """video_c54 2P game_idx=9 (実測 delta_score=30920、真の連鎖数9)。

    chain_count_ocr.py docstring 記載の実障害: この実イベントでテロップの
    連続列方式は digit_2/3/4 の信頼度低下により 3 で頭打ちになった
    (過小評価)。得点逆算は正しく 9 を選べるが、2系統が不一致のため
    fail-safe で chain_count=None を返すべき (数値だけで採否を決めない)。
    """
    telop = ChainCountWindowResult(
        max_chain_count=3,  # 実障害の再現 (連続列の橋渡し失敗)
        samples=(),
        n_hits=9,
        method="monotonic_run",
    )
    result = resolve_chain_count_truth(telop, delta_score=30920)
    assert result.score_chain_count == 9  # 得点逆算単独は真値を正しく当てる
    assert result.telop_chain_count == 3
    assert result.chain_count is None  # だが不一致なので fail-safe
    assert result.reason == "disagree"


# =============================================================================
# fail-safe 経路 (合成データ)
# =============================================================================


def test_resolve_telop_missing() -> None:
    telop = ChainCountWindowResult(max_chain_count=None, samples=(), n_hits=0)
    result = resolve_chain_count_truth(telop, delta_score=_approx_min_chain_score(4))
    assert result.chain_count is None
    assert result.reason == "telop_missing"
    assert result.score_chain_count == 4


def test_resolve_score_missing_when_no_candidate_fits() -> None:
    """どの連鎖数候補の期待得点とも大きく乖離する delta_score は score_missing。"""
    telop = ChainCountWindowResult(
        max_chain_count=3, samples=(), n_hits=1, method="monotonic_run",
    )
    result = resolve_chain_count_truth(telop, delta_score=999_999_999)
    assert result.chain_count is None
    assert result.score_chain_count is None
    assert result.reason == "score_missing"


def test_resolve_both_missing() -> None:
    telop = ChainCountWindowResult(max_chain_count=None, samples=(), n_hits=0)
    result = resolve_chain_count_truth(telop, delta_score=999_999_999)
    assert result.chain_count is None
    assert result.reason == "both_missing"


def test_resolve_agree_requires_exact_match_not_close_value() -> None:
    """テロップと得点逆算が1違い (隣接連鎖数) でも一致扱いにしない (fail-safe厳格性)。"""
    telop = ChainCountWindowResult(
        max_chain_count=4, samples=(), n_hits=1, method="monotonic_run",
    )
    # delta_score は 5連鎖の下限近似ぴったり (score側は5を選ぶはず)
    result = resolve_chain_count_truth(telop, delta_score=_approx_min_chain_score(5))
    assert result.score_chain_count == 5
    assert result.telop_chain_count == 4
    assert result.chain_count is None
    assert result.reason == "disagree"


# =============================================================================
# read_chain_count_truth: end-to-end (実テンプレ + FakeVideoCapture)
# =============================================================================


def test_read_chain_count_truth_end_to_end_agrees_with_real_templates() -> None:
    """実テンプレでテロップ 1→2→3→4→5 を再現し、5連鎖相当の得点と一致させる。"""
    templates = _load_real_templates()
    if not all(d in templates for d in (1, 2, 3, 4, 5)):
        pytest.skip("digit_1〜5 テンプレ未整備")
    ocr = ChainCountOcr.load_default()

    frames = {}
    for i, digit in enumerate([1, 2, 3, 4, 5]):
        f = _make_blank_frame()
        f = _paint_digit_into_board(f, templates[digit], "1P", 30, 200 + digit * 5)
        frames[i * 0.5] = f

    cap = _FakeVideoCapture(frame_by_time_sec=frames, default=_make_blank_frame())
    result = read_chain_count_truth(
        ocr, cap, "1P", t_start=0.0, t_end=2.0,
        delta_score=_approx_min_chain_score(5), sample_interval_sec=0.5,
    )
    assert result.telop_chain_count == 5
    assert result.chain_count == 5
    assert result.reason == "agree"


# =============================================================================
# select_chain_count_high_confidence_band (タスク#7 追加、2026-08-14)
# =============================================================================


def test_high_confidence_band_constants() -> None:
    assert HIGH_CONFIDENCE_SCORE_RATIO_MIN == pytest.approx(0.9)
    assert HIGH_CONFIDENCE_SCORE_RATIO_MAX == pytest.approx(1.1)


def test_high_confidence_band_accepts_exact_ratio_match() -> None:
    """下限近似にぴったり一致する delta_score (比率1.0) は高信頼で採用される。"""
    delta = _approx_min_chain_score(9)  # =27880 (video_c54 2P game_idx=9 相当)
    result = select_chain_count_high_confidence_band(delta)
    assert isinstance(result, HighConfidenceScoreResult)
    assert result.chain_count == 9
    assert result.is_pure_chain_score is True
    assert result.reason == "high_confidence"
    assert result.ratio == pytest.approx(1.0)


def test_high_confidence_band_rejects_non_multiple_of_10() -> None:
    """10の倍数でない delta_score は落下ボーナス混入疑いとして即座に拒否。"""
    delta = _approx_min_chain_score(9) + 1  # 27881、非10倍数
    result = select_chain_count_high_confidence_band(delta)
    assert result.chain_count is None
    assert result.ratio is None
    assert result.is_pure_chain_score is False
    assert result.reason == "contaminated"


def test_high_confidence_band_accepts_real_video_c54_event_via_all_clear_hypothesis() -> None:
    """video_c54 2P game_idx=9 実測 delta_score=30920 (真の連鎖数9) は高信頼で採用される。

    素朴な比率 (30920/27880≈1.109) だけを見ると帯の外に見えるが、
    `score_consistency_ratio` は全消し繰越仮説 (+ALL_CLEAR_BONUS=2100) も
    候補にし対数距離が近い方を採用するため、実際の比率は
    30920/(27880+2100)≈1.031 となり高信頼帯に入る。この実データ整合性は
    `score_consistency_ratio` 側の既存仕様 (allow_all_clear_carryover=True)
    をそのまま利用しているだけであり、本関数側で特別扱いはしていない。
    """
    delta = 30920
    result = select_chain_count_high_confidence_band(delta)
    assert result.chain_count == 9
    assert result.is_pure_chain_score is True
    assert result.reason == "high_confidence"
    assert result.ratio == pytest.approx(30920 / (27880 + 2100), rel=1e-6)
    assert HIGH_CONFIDENCE_SCORE_RATIO_MIN <= result.ratio <= HIGH_CONFIDENCE_SCORE_RATIO_MAX


def test_high_confidence_band_rejects_ratio_outside_tight_band() -> None:
    """10の倍数だが最有力候補の比率が [0.9, 1.1] の外になる合成例。

    5連鎖下限近似 (4840) のちょうど2倍 (9680) は、既存の緩い整合性チェック
    ([0.5, 2.0]) は満たすが、本関数のタイトな高信頼帯では不採用になることを
    確認する (テロップ非依存の独立系統として厳格さを優先する設計)。
    """
    delta = 9680
    result = select_chain_count_high_confidence_band(delta)
    assert result.chain_count is None
    assert result.is_pure_chain_score is True
    assert result.reason == "ratio_out_of_band"
    assert result.ratio is not None
    assert not (HIGH_CONFIDENCE_SCORE_RATIO_MIN <= result.ratio <= HIGH_CONFIDENCE_SCORE_RATIO_MAX)


def test_high_confidence_band_no_candidates() -> None:
    """candidates が空集合の場合は no_candidates (呼び出し側の誤り検知用)。"""
    result = select_chain_count_high_confidence_band(40, candidates=frozenset())
    assert result.chain_count is None
    assert result.ratio is None
    assert result.is_pure_chain_score is True
    assert result.reason == "no_candidates"


def test_high_confidence_band_zero_or_negative_delta_is_contaminated() -> None:
    """delta_score<=0 は連鎖が起きていない (純粋性チェックで False → contaminated)。"""
    result = select_chain_count_high_confidence_band(0)
    assert result.chain_count is None
    assert result.reason == "contaminated"


# =============================================================================
# compute_telop_search_window (続行タスク 2026-08-14、W3「窓ズレ」対処)
# =============================================================================


def test_compute_telop_search_window_real_c13_g12_case() -> None:
    """video_c13 game_idx=12 実測 (before_t=860.1, trigger=866.6) で、実測の
    本物のポップアップ位置 (t≈861.05〜863.9) を窓が確実に含むこと。

    旧実装は MAX_WINDOW_SPAN_SEC=4.5秒 の恣意的キャップで [860.1, 863.1) を
    切り落とし、本物の「1れんさ!」ポップアップ (t≈861.05-861.4) を
    丸ごと見逃していた (実測で確認済み、本モジュール docstring 参照)。
    """
    t_start, t_end = compute_telop_search_window(before_t_sec=860.1, trigger_sec=866.6)
    assert t_start <= 861.05  # 実測の最初のポップアップ開始時刻を含む
    assert t_end >= 863.9  # 実測の最終ステップポップアップ時刻を含む
    assert t_end >= 866.6  # trigger_sec 自体も窓に含む


def test_compute_telop_search_window_applies_small_buffers() -> None:
    t_start, t_end = compute_telop_search_window(before_t_sec=100.0, trigger_sec=110.0)
    assert t_start == pytest.approx(100.0 - TELOP_SEARCH_PRE_BUFFER_SEC)
    assert t_end == pytest.approx(110.0 + TELOP_SEARCH_POST_BUFFER_SEC)


def test_compute_telop_search_window_no_artificial_span_cap() -> None:
    """区間が長い (実測最大 9.63秒 の offset_settle 相当) でも切り詰めないこと。

    旧実装のバグ (実測に基づかない MAX_WINDOW_SPAN_SEC) の再発防止テスト。
    """
    t_start, t_end = compute_telop_search_window(before_t_sec=0.0, trigger_sec=20.0)
    assert t_end - t_start >= 20.0


def test_compute_telop_search_window_guards_against_inverted_input() -> None:
    """trigger_sec < before_t_sec という異常入力でも t_end < t_start にならない。"""
    t_start, t_end = compute_telop_search_window(before_t_sec=50.0, trigger_sec=10.0)
    assert t_end >= t_start


def test_read_chain_count_truth_no_popup_returns_unknown() -> None:
    """ポップアップが一度も出ない window では不明 (both_missing に近い経路)。"""
    ocr = ChainCountOcr.load_default()
    cap = _FakeVideoCapture(frame_by_time_sec={}, default=_make_blank_frame())
    result = read_chain_count_truth(
        ocr, cap, "1P", t_start=0.0, t_end=1.0, delta_score=999_999_999,
        sample_interval_sec=0.5,
    )
    assert result.chain_count is None
    assert result.reason == "both_missing"
