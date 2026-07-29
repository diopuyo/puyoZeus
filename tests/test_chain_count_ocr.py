"""src/chain_count_ocr.py のテスト。

「N れんさ!」ポップアップ OCR を合成画像 + 実テンプレで検証する。
表示位置が可変な仕様のため、ROI 内の複数位置にテンプレを貼って
位置非依存性 (盤面全体スキャン) を確認するのが本テストの要点。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.chain_count_ocr import (
    CHAIN_COUNT_MAX,
    CHAIN_COUNT_MIN,
    CHAIN_DIGIT_HEIGHT,
    CHAIN_DIGIT_LABELS,
    CHAIN_DIGIT_WIDTH,
    EXPECTED_FRAME_SHAPE,
    ChainCountOcr,
    ChainCountReadResult,
    ChainCountWindowResult,
    _aggregate_window_samples,
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
    return np.full((h, w, 3), 40, dtype=np.uint8)  # 盤面の暗い背景を模した一様色


def _paint_digit_into_board(
    frame: np.ndarray, digit_img: np.ndarray, side: str,
    offset_x: int, offset_y: int,
) -> np.ndarray:
    """盤面 ROI 内の任意位置にテンプレを貼り付ける (位置可変性の再現)。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    ax = region.x + offset_x
    ay = region.y + offset_y
    h, w = digit_img.shape[:2]
    frame[ay:ay + h, ax:ax + w] = digit_img
    return frame


class _FakeVideoCapture:
    """cv2.VideoCapture の set()/read() のみを模した簡易フェイク。

    read_max_in_window() は set(CAP_PROP_POS_MSEC, ...) → read() の順で
    呼ばれる想定のため、set() で指定した時刻に応じてフレームを切り替える。
    """

    def __init__(self, frame_by_time_sec: dict[float, np.ndarray], default: np.ndarray) -> None:
        self._frame_by_time = frame_by_time_sec
        self._default = default
        self._cur_time_ms: float = 0.0

    def set(self, prop: int, value: float) -> None:
        self._cur_time_ms = value

    def read(self) -> tuple[bool, np.ndarray]:
        t_sec = round(self._cur_time_ms / 1000.0, 2)
        # 最も近い登録時刻のフレームを返す (テスト用の簡易実装)
        if not self._frame_by_time:
            return True, self._default
        closest = min(self._frame_by_time, key=lambda k: abs(k - t_sec))
        return True, self._frame_by_time[closest]


# =============================================================================
# 定数・基本形状
# =============================================================================


def test_chain_count_ocr_constants() -> None:
    assert CHAIN_COUNT_MIN == 1
    assert CHAIN_COUNT_MAX >= 4  # video_c54 で 4連鎖まで実測確認済み
    assert len(CHAIN_DIGIT_LABELS) == CHAIN_COUNT_MAX - CHAIN_COUNT_MIN + 1
    assert CHAIN_DIGIT_HEIGHT > 0 and CHAIN_DIGIT_WIDTH > 0


def test_chain_count_ocr_no_templates_returns_none() -> None:
    """テンプレを 1 つも渡さない場合は常に None。"""
    ocr = ChainCountOcr(templates={})
    frame = _make_blank_frame()
    res = ocr.read_side(frame, "1P")
    assert isinstance(res, ChainCountReadResult)
    assert res.chain_count is None


def test_chain_count_ocr_invalid_frame_shape() -> None:
    """形状不正フレームでは None を返す (クラッシュしない)。"""
    ocr = ChainCountOcr.load_default()
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    res = ocr.read_side(bad, "1P")
    assert res.chain_count is None


def test_chain_count_ocr_load_default_present_digits() -> None:
    """1-4 は実テンプレが配置済み。5-9 欠損でも load 自体は失敗しない。"""
    ocr = ChainCountOcr.load_default()
    assert ocr is not None


# =============================================================================
# 位置可変性: ROI 内の異なる座標でも読み取れることを確認 (本モジュールの要点)
# =============================================================================


@pytest.mark.parametrize("digit", [1, 2, 3, 4])
@pytest.mark.parametrize("offset", [(30, 200), (150, 450)])
def test_chain_count_ocr_reads_digit_at_various_positions(
    digit: int, offset: tuple[int, int],
) -> None:
    templates = _load_real_templates()
    if digit not in templates:
        pytest.skip(f"digit_{digit} テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    tpl = templates[digit]
    offset_x, offset_y = offset
    frame = _paint_digit_into_board(frame, tpl, "1P", offset_x, offset_y)
    res = ocr.read_side(frame, "1P")
    assert res.chain_count == digit


def test_chain_count_ocr_side_2p_independent_of_1p() -> None:
    """2P 側 ROI に貼っても 1P 側からは検出されない (ROI 分離の確認)。"""
    templates = _load_real_templates()
    if 2 not in templates:
        pytest.skip("digit_2 テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_digit_into_board(frame, templates[2], "2P", 100, 300)
    res_2p = ocr.read_side(frame, "2P")
    res_1p = ocr.read_side(frame, "1P")
    assert res_2p.chain_count == 2
    assert res_1p.chain_count is None


def test_chain_count_ocr_no_popup_returns_none() -> None:
    """ポップアップが出ていない (無地) 盤面では None を返す。"""
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    res = ocr.read_side(frame, "1P")
    assert res.chain_count is None


# =============================================================================
# window内 最大値集計 (純粋関数 + read_max_in_window)
# =============================================================================


def test_aggregate_window_samples_takes_max() -> None:
    """連鎖ステップが進むたびに値が増える想定 → window内最大値を採用。"""
    samples = [
        ChainCountReadResult(1, 0.9, (0, 0)),
        ChainCountReadResult(None, 0.1, None),
        ChainCountReadResult(2, 0.85, (5, 5)),
        ChainCountReadResult(4, 0.8, (10, 10)),
        ChainCountReadResult(3, 0.7, (1, 1)),  # 揺り戻し (誤検出等) があっても max を採用
    ]
    result = _aggregate_window_samples(samples)
    assert isinstance(result, ChainCountWindowResult)
    assert result.max_chain_count == 4
    assert result.n_hits == 4


def test_aggregate_window_samples_all_none() -> None:
    samples = [ChainCountReadResult(None, 0.0, None) for _ in range(5)]
    result = _aggregate_window_samples(samples)
    assert result.max_chain_count is None
    assert result.n_hits == 0


def test_aggregate_window_samples_empty() -> None:
    result = _aggregate_window_samples([])
    assert result.max_chain_count is None
    assert result.n_hits == 0


def test_read_max_in_window_picks_max_across_fake_frames() -> None:
    """複数時刻のフレームを模した FakeVideoCapture で window 内最大値を取得。"""
    templates = _load_real_templates()
    if not all(d in templates for d in (1, 2, 3)):
        pytest.skip("digit_1/2/3 テンプレ未整備")
    ocr = ChainCountOcr.load_default()

    frame_t0 = _make_blank_frame()
    frame_t0 = _paint_digit_into_board(frame_t0, templates[1], "1P", 30, 200)
    frame_t1 = _make_blank_frame()
    frame_t1 = _paint_digit_into_board(frame_t1, templates[2], "1P", 30, 340)
    frame_t2 = _make_blank_frame()
    frame_t2 = _paint_digit_into_board(frame_t2, templates[3], "1P", 30, 480)

    cap = _FakeVideoCapture(
        frame_by_time_sec={0.0: frame_t0, 0.5: frame_t1, 1.0: frame_t2},
        default=_make_blank_frame(),
    )
    result = ocr.read_max_in_window(
        cap, "1P", t_start=0.0, t_end=1.0, sample_interval_sec=0.5,
    )
    assert result.max_chain_count == 3
    assert result.n_hits == 3


def test_read_max_in_window_invalid_range_returns_none() -> None:
    ocr = ChainCountOcr.load_default()
    cap = _FakeVideoCapture(frame_by_time_sec={}, default=_make_blank_frame())
    result = ocr.read_max_in_window(cap, "1P", t_start=2.0, t_end=1.0)
    assert result.max_chain_count is None
    assert result.samples == ()
