"""frame_reader.read_frames_sequential テスト (Z-3C)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.frame_reader import FrameSample, read_frames_sequential


def _find_test_video() -> Path | None:
    """既存の test 動画を探す。data/frames/video_18.mp4 等。"""
    root = Path(__file__).resolve().parent.parent
    for vid in (1, 2, 3, 18):
        p = root / f"data/frames/video_{vid:02d}.mp4"
        if p.exists():
            return p
    return None


def test_returns_correct_count() -> None:
    """time list の長さと出力数が一致。"""
    video = _find_test_video()
    if video is None:
        # CI で動画ファイル無しならスキップ
        return
    times = [10.0, 10.1, 10.2, 10.3, 10.4]
    samples = read_frames_sequential(str(video), times)
    assert len(samples) == 5


def test_frame_samples_have_target_size() -> None:
    """出力 frame は 1920x1080 にリサイズされている。"""
    video = _find_test_video()
    if video is None:
        return
    times = [10.0, 10.5, 11.0]
    samples = read_frames_sequential(str(video), times)
    for s in samples:
        if s is not None:
            assert s.frame.shape[:2] == (1080, 1920)


def test_empty_times_returns_empty() -> None:
    video = _find_test_video()
    if video is None:
        return
    samples = read_frames_sequential(str(video), [])
    assert samples == []


def test_invalid_video_returns_none_list() -> None:
    samples = read_frames_sequential("/nonexistent/path.mp4", [0.0, 0.5])
    assert samples == [None, None]
