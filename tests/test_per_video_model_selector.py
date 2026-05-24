"""per_video_model_selector テスト。"""
from __future__ import annotations

from pathlib import Path

from src.per_video_model_selector import (
    V17B_BEST_VIDEOS, select_model_for_video,
)


def test_v10_returns_v17b_if_exists() -> None:
    """v10 は v17b best、ファイル存在で v17b を返す。"""
    model_path = select_model_for_video("data/frames/video_10.mp4")
    if Path("models/cnn_phase_u_v17b.pt").exists():
        assert "v17b" in model_path
    else:
        assert "v16" in model_path


def test_v12_returns_v16() -> None:
    """v12 は v17b で大幅悪化、v16 を返す。"""
    model_path = select_model_for_video("data/frames/video_12.mp4")
    assert "v16" in model_path


def test_v13_returns_v16() -> None:
    model_path = select_model_for_video("data/frames/video_13.mp4")
    assert "v16" in model_path


def test_unknown_video_returns_v16() -> None:
    """未知動画 (mapping にない) は default v16。"""
    model_path = select_model_for_video("data/frames/video_99.mp4")
    assert "v16" in model_path


def test_v17b_best_videos_set() -> None:
    """V17B_BEST_VIDEOS に予期した動画が含まれている。"""
    assert "video_10" in V17B_BEST_VIDEOS
    assert "video_11" in V17B_BEST_VIDEOS
    assert "video_19" in V17B_BEST_VIDEOS
    assert "video_12" not in V17B_BEST_VIDEOS  # v17b で悪化
    assert "video_13" not in V17B_BEST_VIDEOS
