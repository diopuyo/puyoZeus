"""V3.2 MatchEndDetector テスト。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.match_end_detector import (
    DEFAULT_LOCKDOWN_SEC,
    MatchEndDetectionResult,
    MatchEndDetector,
)


def _load_frame(path: str) -> np.ndarray:
    """フレーム画像を 1080p 形式でロード。"""
    fr = cv2.imread(path)
    assert fr is not None, f"failed to load: {path}"
    if fr.shape[:2] != (1080, 1920):
        fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
    return fr


def test_detector_loads_default_templates() -> None:
    """既定ディレクトリから match_end_*.png をロード。"""
    detector = MatchEndDetector.load_default()
    # match_end_yatta.png と match_end_batan.png が存在するはず
    assert len(detector._templates) >= 2
    assert "match_end_yatta" in detector._templates
    assert "match_end_batan" in detector._templates


def test_initial_state_not_locked() -> None:
    """初期状態 (まだ何も検出していない) はロック中ではない。"""
    detector = MatchEndDetector.load_default()
    assert detector.is_locked(0.0) is False
    assert detector.is_locked(100.0) is False
    assert detector.last_detected_t is None


def test_blank_frame_no_detection() -> None:
    """全黒フレームではテンプレートマッチ低 → 検出なし。"""
    detector = MatchEndDetector.load_default()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.detected is False


def test_detection_on_actual_match_end_frame() -> None:
    """実際の試合終了フレーム (v01_m1 t=253) で検出される。"""
    path = "data/verify/match_end_frames/scan_v01_m1_t253.png"
    if not Path(path).exists():
        pytest.skip(f"frame not available: {path}")
    detector = MatchEndDetector.load_default()
    frame = _load_frame(path)
    result = detector.detect(frame)
    # テンプレ自体がこのフレームから切り出されているので score 高いはず
    assert result.detected is True
    assert result.score >= 0.55


def test_lockdown_after_detection() -> None:
    """検出時刻から lockdown_sec の間ロック中。"""
    path = "data/verify/match_end_frames/scan_v01_m1_t253.png"
    if not Path(path).exists():
        pytest.skip("frame not available")
    detector = MatchEndDetector.load_default(lockdown_sec=5.0)
    frame = _load_frame(path)

    # t=253 で検出 → ロック開始
    locked_at_detection = detector.update(frame, 253.0)
    assert locked_at_detection is True
    # t=255 (2 秒後) もロック中
    assert detector.is_locked(255.0) is True
    # t=257.99 (4.99 秒後) もロック中
    assert detector.is_locked(257.99) is True
    # t=258.01 (5.01 秒後) はロック解除
    assert detector.is_locked(258.01) is False


def test_lockdown_extended_by_repeated_detection() -> None:
    """連続検出されるとロック開始時刻が更新される (リセットして延長)。"""
    path = "data/verify/match_end_frames/scan_v01_m1_t253.png"
    if not Path(path).exists():
        pytest.skip("frame not available")
    detector = MatchEndDetector.load_default(lockdown_sec=3.0)
    frame = _load_frame(path)
    detector.update(frame, 253.0)
    # t=255 でも再検出 → last_detected_t=255 に更新
    detector.update(frame, 255.0)
    # t=257 (255 から 2 秒) はまだロック中
    assert detector.is_locked(257.0) is True
    # t=258.01 (255 から 3.01 秒) はロック解除
    assert detector.is_locked(258.01) is False


def test_reset_clears_lockdown() -> None:
    """reset() でタイマーが消える。"""
    path = "data/verify/match_end_frames/scan_v01_m1_t253.png"
    if not Path(path).exists():
        pytest.skip("frame not available")
    detector = MatchEndDetector.load_default()
    frame = _load_frame(path)
    detector.update(frame, 253.0)
    assert detector.is_locked(253.5) is True
    detector.reset()
    assert detector.is_locked(253.5) is False
    assert detector.last_detected_t is None


def test_no_false_positive_on_in_match_frame() -> None:
    """試合中の通常フレーム (テロップ常時表示) では検出されない。

    既存の match_end_frames には試合中の roi_check_video_01.png 等を
    使わず、既知の終了フレームの 10 秒前 (t=243 付近) で検証する。
    """
    cap = cv2.VideoCapture("data/frames/video_01.mp4")
    if not cap.isOpened():
        pytest.skip("video not available")
    # 試合中 (m1 開始 200s 〜 終了 256s の中盤、t=230)
    cap.set(cv2.CAP_PROP_POS_MSEC, 230 * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        pytest.skip("frame fetch failed")
    if fr.shape[:2] != (1080, 1920):
        fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
    detector = MatchEndDetector.load_default()
    result = detector.detect(fr)
    # 試合中の中盤フレームではやった/ばたんきゅーは表示されないので未検出が期待
    assert result.detected is False, (
        f"unexpected detection on mid-match frame: score={result.score}"
    )


def test_detection_result_dataclass() -> None:
    """MatchEndDetectionResult のフィールド確認。"""
    r = MatchEndDetectionResult(detected=False, template_name=None, score=0.0)
    assert r.detected is False
    r2 = MatchEndDetectionResult(
        detected=True, template_name="match_end_yatta", score=0.85,
    )
    assert r2.detected is True
    assert r2.template_name == "match_end_yatta"
