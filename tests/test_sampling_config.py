"""src/sampling_config のテスト。"""
from __future__ import annotations

from src.sampling_config import (
    BOARD_INTERVAL_SEC,
    EVAL_FRAME_RATIO,
    EVAL_INTERVAL_SEC,
    STABLE_FRAME_COUNT,
    STABLE_FRAME_INTERVAL_SEC,
    board_sample_times,
    eval_sample_times,
)


def test_intervals_consistent() -> None:
    """評価間隔は盤面サンプリングの ratio 倍。"""
    assert abs(EVAL_INTERVAL_SEC - BOARD_INTERVAL_SEC * EVAL_FRAME_RATIO) < 1e-6


def test_board_sample_times_basic() -> None:
    times = board_sample_times(0.0, 1.0)
    # 0.0, 0.2, 0.4, 0.6, 0.8 (1.0 は < 条件で除外)
    assert len(times) == 5
    assert abs(times[0] - 0.0) < 1e-6
    assert abs(times[1] - 0.2) < 1e-6
    assert abs(times[-1] - 0.8) < 1e-6


def test_eval_sample_times_basic() -> None:
    times = eval_sample_times(0.0, 2.0)
    # 0.0, 0.6, 1.2, 1.8
    assert len(times) == 4


def test_stable_frame_constants() -> None:
    """安定判定: 0.2s × 2 連続。"""
    assert STABLE_FRAME_COUNT == 2
    assert abs(STABLE_FRAME_INTERVAL_SEC - 0.2) < 1e-6
