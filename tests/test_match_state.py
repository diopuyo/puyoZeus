"""MatchStateDetector のテスト。

既知の in-match / not-in-match フレームで判定精度をテストする。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from src.match_state import MatchState, MatchStateDetector, IN_MATCH_V_MAX


# テスト対象フレーム（video_01 / video_02）
IN_MATCH_FRAMES = [
    "data/frames/sample/frame_0300s.png",
    "data/frames/sample/frame_0600s.png",
    "data/frames/sample/frame_0900s.png",
    "data/frames/sample/frame_1500s.png",
    "data/frames/sample/frame_2100s.png",
    "data/frames/sample/frame_2700s.png",
    "data/frames/sample/frame_3200s.png",
]

NOT_IN_MATCH_FRAMES = [
    "tests/fixtures/frames/not_match_browser_0050.png",  # ブラウザ画面
    "tests/fixtures/frames/not_match_vs_0170.png",       # VS 画面
]


def _load_if_exists(path: str):
    p = Path(path)
    if not p.exists():
        pytest.skip(f"テスト用フレームが存在しない: {p}")
    frame = cv2.imread(str(p))
    if frame is None:
        pytest.skip(f"フレーム読み込み失敗: {p}")
    return frame


@pytest.fixture(scope="module")
def detector() -> MatchStateDetector:
    calib = Path("models/calibration_video01.json")
    if not calib.exists():
        pytest.skip("calibration ファイルがない")
    return MatchStateDetector.load_default(calib)


@pytest.mark.parametrize("path", IN_MATCH_FRAMES)
def test_in_match_frames(detector: MatchStateDetector, path: str) -> None:
    frame = _load_if_exists(path)
    result = detector.detect(frame)
    assert result.state == MatchState.IN_MATCH, (
        f"{path}: 試合中と判定されるはずだが {result.state}  "
        f"bg_value={result.bg_value:.1f} (threshold={IN_MATCH_V_MAX})"
    )


@pytest.mark.parametrize("path", NOT_IN_MATCH_FRAMES)
def test_not_in_match_frames(detector: MatchStateDetector, path: str) -> None:
    frame = _load_if_exists(path)
    result = detector.detect(frame)
    assert result.state == MatchState.NOT_IN_MATCH, (
        f"{path}: 非試合と判定されるはずだが {result.state}  "
        f"bg_value={result.bg_value:.1f} (threshold={IN_MATCH_V_MAX})"
    )


def test_threshold_separation(detector: MatchStateDetector) -> None:
    """in-match と not-in-match の bg_value が閾値で明確に分離するか確認。"""
    in_vals: list[float] = []
    out_vals: list[float] = []
    for path in IN_MATCH_FRAMES:
        p = Path(path)
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        in_vals.append(detector.detect(frame).bg_value)
    for path in NOT_IN_MATCH_FRAMES:
        p = Path(path)
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        out_vals.append(detector.detect(frame).bg_value)
    if not in_vals or not out_vals:
        pytest.skip("十分なデータがない")
    # 最大 in-match < 最小 not-in-match で完全分離を要求
    max_in = max(in_vals)
    min_out = min(out_vals)
    assert max_in < min_out, (
        f"分離失敗: max(in_match)={max_in:.1f} >= min(not_in_match)={min_out:.1f}"
    )
    assert max_in < IN_MATCH_V_MAX < min_out, (
        f"閾値 {IN_MATCH_V_MAX} が分離点でない: max_in={max_in:.1f}, min_out={min_out:.1f}"
    )
