"""src/background_fingerprint.py のテスト (Phase T サイクル 1)。"""
from __future__ import annotations

import numpy as np
import pytest

from src.background_fingerprint import (
    DEFAULT_EMPTY_HSV_DISTANCE,
    BackgroundFingerprint,
    CellFingerprint,
    capture_pair_fingerprint,
    capture_pair_robust,
    capture_robust_fingerprint,
    is_empty_by_fp,
)


def _solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    return np.full((1080, 1920, 3), bgr, dtype=np.uint8)


def test_cell_fingerprint_distance_zero() -> None:
    a = CellFingerprint(60, 100, 200)
    assert a.distance_to(a) == 0


def test_cell_fingerprint_distance_h_circular() -> None:
    """H=5 と H=175 は実距離 10 (循環)。"""
    a = CellFingerprint(5, 100, 200)
    b = CellFingerprint(175, 100, 200)
    # H 距離 10 × 0.5 = 5
    assert a.distance_to(b) == 5


def test_capture_uniform_frame() -> None:
    """単色フレーム → 全セル同じ FP。"""
    frame = _solid_frame((40, 80, 200))  # BGR、HSV では特定値
    fp = BackgroundFingerprint.capture(frame, 282, 160, 384, 720)
    # 全セルで同じ HSV になる
    cells = fp.cells
    base = cells[0][0]
    for row in cells:
        for c in row:
            assert c.distance_to(base) == 0


def test_capture_pair_returns_two() -> None:
    frame = _solid_frame((30, 30, 30))
    fp1, fp2 = capture_pair_fingerprint(
        frame, (282, 160, 384, 720), (1258, 160, 384, 720),
    )
    assert isinstance(fp1, BackgroundFingerprint)
    assert isinstance(fp2, BackgroundFingerprint)


def test_is_empty_by_fp_same_returns_true() -> None:
    fp = CellFingerprint(60, 100, 200)
    assert is_empty_by_fp(fp, fp)


def test_is_empty_by_fp_far_returns_false() -> None:
    fp_bg = CellFingerprint(60, 100, 200)
    fp_puyo = CellFingerprint(0, 220, 230)
    assert not is_empty_by_fp(fp_puyo, fp_bg)


def test_is_empty_by_fp_threshold() -> None:
    fp_bg = CellFingerprint(60, 100, 200)
    fp_close = CellFingerprint(60, 110, 200)  # S+10 → 距離 10
    assert is_empty_by_fp(
        fp_close, fp_bg, threshold=DEFAULT_EMPTY_HSV_DISTANCE,
    )
    assert not is_empty_by_fp(fp_close, fp_bg, threshold=5.0)


def test_cell_at_out_of_range() -> None:
    frame = _solid_frame((30, 30, 30))
    fp = BackgroundFingerprint.capture(frame, 282, 160, 384, 720)
    assert fp.cell_at(-1, 0) == CellFingerprint(0, 0, 0)
    assert fp.cell_at(0, 100) == CellFingerprint(0, 0, 0)


def test_invalid_frame_returns_default() -> None:
    fp = BackgroundFingerprint.capture(None, 0, 0, 384, 720)  # type: ignore
    assert fp.cells[0][0] == CellFingerprint(0, 0, 0)


# ============================
# T-v2-N: capture_robust_fingerprint
# ============================


def test_robust_fp_empty_frames_returns_zero() -> None:
    """空リスト → 全 0 セル。"""
    fp = capture_robust_fingerprint([], 282, 160, 384, 720)
    assert fp.cells[0][0] == CellFingerprint(0, 0, 0)


def test_robust_fp_single_frame_matches_capture() -> None:
    """1 フレームのみ → BackgroundFingerprint.capture と等価。"""
    f = _solid_frame((30, 30, 30))
    fp_robust = capture_robust_fingerprint([f], 282, 160, 384, 720)
    fp_single = BackgroundFingerprint.capture(f, 282, 160, 384, 720)
    assert fp_robust.cells[0][0] == fp_single.cells[0][0]


def test_robust_fp_median_drops_outlier() -> None:
    """5 フレーム中 1 フレームだけ大きな外れ値 → median で除外。"""
    normal = _solid_frame((30, 30, 30))    # 暗い
    outlier = _solid_frame((200, 200, 200))  # 明るい (キャラの瞬間動き想定)
    frames = [normal, normal, outlier, normal, normal]
    fp = capture_robust_fingerprint(frames, 282, 160, 384, 720)
    # median は normal 側
    cell = fp.cell_at(0, 0)
    assert cell.v < 60  # outlier 除外できれば V≈30


def test_robust_fp_handles_invalid_frame() -> None:
    """None フレームを混ぜても落ちない。"""
    f = _solid_frame((50, 50, 50))
    frames = [f, None, f]  # type: ignore[list-item]
    fp = capture_robust_fingerprint(frames, 282, 160, 384, 720)
    # 残り 2 フレームの median が反映される
    cell = fp.cell_at(0, 0)
    assert 0 <= cell.v < 100


def test_capture_pair_robust_returns_two_fps() -> None:
    """1P/2P 両方のロバスト FP を返す。"""
    f = _solid_frame((30, 30, 30))
    fp1, fp2 = capture_pair_robust(
        [f, f, f],
        (282, 160, 384, 720),
        (1258, 160, 384, 720),
    )
    assert isinstance(fp1, BackgroundFingerprint)
    assert isinstance(fp2, BackgroundFingerprint)


# ============================
# 案 P2: detect_highlight_blob テスト
# ============================

from src.background_fingerprint import (  # noqa: E402
    HIGHLIGHT_MIN_PIXEL_RATIO,
    HIGHLIGHT_S_MAX,
    HIGHLIGHT_V_MIN,
    detect_highlight_blob,
)


def _make_patch_hsv(
    h: int, s: int, v: int, h_size: int = 20, w_size: int = 20,
) -> np.ndarray:
    """指定 HSV で均一なパッチを生成 (uint8)。"""
    patch = np.zeros((h_size, w_size, 3), dtype=np.uint8)
    patch[:, :, 0] = h
    patch[:, :, 1] = s
    patch[:, :, 2] = v
    return patch


def test_highlight_detected_white_blob() -> None:
    """上部帯域に白 blob (V=240, S=20) が十分あれば True。"""
    # 20×20 パッチ全体が白 → 上部帯域 (15%〜55% = 行3〜11) が白 → min_ratio を超える
    patch = _make_patch_hsv(h=0, s=20, v=240)
    assert detect_highlight_blob(patch) is True


def test_highlight_not_detected_dark() -> None:
    """全セルが暗い (V=30) → 白 blob なし → False。"""
    patch = _make_patch_hsv(h=0, s=0, v=30)
    assert detect_highlight_blob(patch) is False


def test_highlight_not_detected_colored() -> None:
    """彩度が高い色 (S=200, V=240) → 白判定不可 → False。"""
    patch = _make_patch_hsv(h=60, s=200, v=240)
    assert detect_highlight_blob(patch) is False


def test_highlight_boundary_v_min() -> None:
    """V = HIGHLIGHT_V_MIN (境界値) かつ S=0 → True (以上は含む)。"""
    patch = _make_patch_hsv(h=0, s=0, v=HIGHLIGHT_V_MIN)
    assert detect_highlight_blob(patch) is True


def test_highlight_boundary_s_max() -> None:
    """S = HIGHLIGHT_S_MAX (境界値) かつ V=255 → True (以下は含む)。"""
    patch = _make_patch_hsv(h=0, s=HIGHLIGHT_S_MAX, v=255)
    assert detect_highlight_blob(patch) is True


def test_highlight_uniform_zero() -> None:
    """全ゼロパッチ (V=0, S=0) → V < HIGHLIGHT_V_MIN → False。"""
    patch = np.zeros((20, 20, 3), dtype=np.uint8)
    assert detect_highlight_blob(patch) is False
