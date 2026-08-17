"""src.board_motion のユニットテスト (2026-08-18 新設、(d) STABLE持続確認)。"""
from __future__ import annotations

import numpy as np
import pytest

from src.board_motion import (
    STABLE_PERSISTENCE_DIFF_THRESHOLD,
    STABLE_PERSISTENCE_WINDOW_SEC,
    board_roi_gray,
    column_diffs,
    frame_diff_mean,
    is_raw_pixel_stable,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION


def _dummy_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# board_roi_gray
# ============================


def test_board_roi_gray_shape_matches_region_1p() -> None:
    frame = _dummy_frame()
    gray = board_roi_gray(frame, "1P")
    assert gray.shape == (DEFAULT_P1_REGION.height, DEFAULT_P1_REGION.width)


def test_board_roi_gray_shape_matches_region_2p() -> None:
    frame = _dummy_frame()
    gray = board_roi_gray(frame, "2P")
    assert gray.shape == (DEFAULT_P2_REGION.height, DEFAULT_P2_REGION.width)


def test_board_roi_gray_returns_2d_array() -> None:
    """grayscale 変換のため channel 軸が無いこと (BGR2GRAY の確認)。"""
    frame = _dummy_frame()
    gray = board_roi_gray(frame, "1P")
    assert gray.ndim == 2


# ============================
# column_diffs / frame_diff_mean (診断スクリプトとの数値一致回帰テスト)
# ============================


def _reference_column_diffs(
    prev_gray: np.ndarray, cur_gray: np.ndarray, n_cols: int = 6,
) -> "list[float]":
    """scripts/_diag_general_chain_contamination_2026-08-17.py:166-175 の
    _column_diffs をそのまま複製した参照実装 (回帰テスト用、意図的な複製)。
    """
    h, w = cur_gray.shape
    col_w = w / n_cols
    diff = np.abs(cur_gray.astype(np.int16) - prev_gray.astype(np.int16))
    out = []
    for i in range(n_cols):
        x1, x2 = int(i * col_w), int((i + 1) * col_w)
        out.append(float(diff[:, x1:x2].mean()))
    return out


def test_column_diffs_zero_when_identical() -> None:
    gray = np.full((100, 120), 128, dtype=np.uint8)
    diffs = column_diffs(gray, gray)
    assert diffs == [0.0] * 6


def test_column_diffs_matches_diag_script_reference() -> None:
    """診断スクリプトの _column_diffs と数値的に完全一致すること。"""
    rng = np.random.default_rng(42)
    prev = rng.integers(0, 256, size=(80, 120), dtype=np.uint8)
    cur = rng.integers(0, 256, size=(80, 120), dtype=np.uint8)
    got = column_diffs(prev, cur)
    expected = _reference_column_diffs(prev, cur)
    assert got == pytest.approx(expected)


def test_column_diffs_detects_localized_spike() -> None:
    """1 列だけ大きく変化した場合、その列の diff だけが大きくなる
    (診断データの「col3のみ単独diff spike」等と同型の挙動確認)。
    """
    prev = np.full((60, 120), 100, dtype=np.uint8)
    cur = prev.copy()
    cur[:, 60:80] = 200  # 6等分 (各20px幅) の4列目 (index 3) だけ変化
    diffs = column_diffs(prev, cur)
    assert diffs[3] > 50.0
    for i in (0, 1, 2, 4, 5):
        assert diffs[i] == 0.0


def test_frame_diff_mean_is_average_of_column_diffs() -> None:
    rng = np.random.default_rng(7)
    prev = rng.integers(0, 256, size=(80, 120), dtype=np.uint8)
    cur = rng.integers(0, 256, size=(80, 120), dtype=np.uint8)
    mean_val = frame_diff_mean(prev, cur)
    assert mean_val == pytest.approx(float(np.mean(column_diffs(prev, cur))))


def test_frame_diff_mean_zero_when_static() -> None:
    gray = np.full((80, 120), 50, dtype=np.uint8)
    assert frame_diff_mean(gray, gray) == 0.0


# ============================
# is_raw_pixel_stable
# ============================


def test_is_raw_pixel_stable_empty_history_returns_true() -> None:
    """観測が無い (直前フレーム無し等) 場合は保守的に True (収集継続)。"""
    assert is_raw_pixel_stable([]) is True


def test_is_raw_pixel_stable_all_below_threshold() -> None:
    assert is_raw_pixel_stable([0.1, 0.2, 0.858]) is True


def test_is_raw_pixel_stable_one_above_threshold() -> None:
    """実測分離ギャップ (0.858 vs 1.07) の境界例。"""
    assert is_raw_pixel_stable([0.1, 0.2, 1.07]) is False


def test_is_raw_pixel_stable_contamination_evidence_values() -> None:
    """実測 (classification_corrected_2026-08-17.json) の混入 diff 値
    (030=試合外を除く5件) は全て閾値超過で不安定と判定される。
    """
    contaminated_diffs = [4.711, 4.196, 1.417, 1.146, 1.07]
    for d in contaminated_diffs:
        assert is_raw_pixel_stable([d]) is False, f"diff={d} は不安定判定されるべき"


def test_is_raw_pixel_stable_clean_evidence_value() -> None:
    """実測 (report.md) の綺麗な21枚の min_diff_near 最大値は安定判定される。"""
    assert is_raw_pixel_stable([0.858]) is True


def test_is_raw_pixel_stable_custom_threshold() -> None:
    assert is_raw_pixel_stable([0.5], diff_threshold=0.3) is False
    assert is_raw_pixel_stable([0.5], diff_threshold=0.6) is True


def test_constants_are_positive() -> None:
    assert STABLE_PERSISTENCE_DIFF_THRESHOLD > 0.0
    assert STABLE_PERSISTENCE_WINDOW_SEC > 0.0
