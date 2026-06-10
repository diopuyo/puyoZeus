"""scripts.generate_training_dataset_v2 のスモークテスト。"""

from __future__ import annotations

import pytest

from scripts.old.generate_training_dataset import MatchMeta
from scripts.old.generate_training_dataset_v2 import (
    DEFAULT_TIME_PHASES_V2,
    PHASE_DEFINITIONS,
    TIME_PHASE_END_M5,
    TIME_PHASE_END_M15,
    TIME_PHASE_MID_M30,
    TIME_PHASE_MID_P30,
    TIME_PHASE_MIDPOINT,
    TIME_PHASE_START_0,
    TIME_PHASE_START_15,
    TIME_PHASE_START_30,
    compute_sample_time_v2,
)


def _meta(start: float = 100.0, end: float = 200.0) -> MatchMeta:
    return MatchMeta(
        video_id="01", match_idx=1,
        start_sec=start, end_sec=end, winner="1P",
    )


def test_default_phases_count_is_ten() -> None:
    """v2 のデフォルト時刻は 10 個。"""
    assert len(DEFAULT_TIME_PHASES_V2) == 10
    # 全 phase が定義に存在
    for p in DEFAULT_TIME_PHASES_V2:
        assert p in PHASE_DEFINITIONS


def test_compute_sample_time_v2_anchors() -> None:
    """各 phase が想定通りの秒数を返すこと。"""
    m = _meta(100.0, 200.0)
    midpoint = 150.0
    # start anchor
    assert compute_sample_time_v2(m, TIME_PHASE_START_0) == pytest.approx(101.0)
    assert compute_sample_time_v2(m, TIME_PHASE_START_15) == pytest.approx(115.0)
    assert compute_sample_time_v2(m, TIME_PHASE_START_30) == pytest.approx(130.0)
    # mid anchor
    assert compute_sample_time_v2(m, TIME_PHASE_MID_M30) == pytest.approx(120.0)
    assert compute_sample_time_v2(m, TIME_PHASE_MIDPOINT) == pytest.approx(midpoint)
    assert compute_sample_time_v2(m, TIME_PHASE_MID_P30) == pytest.approx(180.0)
    # end anchor
    assert compute_sample_time_v2(m, TIME_PHASE_END_M5) == pytest.approx(195.0)
    assert compute_sample_time_v2(m, TIME_PHASE_END_M15) == pytest.approx(185.0)


def test_compute_sample_time_v2_clamps_to_match_range() -> None:
    """試合範囲を超える時刻は端点に丸められる。"""
    short = _meta(50.0, 60.0)  # 10 秒試合 (実際は MIN_DURATION でフィルタ)
    # mid_plus_30 は end を超過 → end-0.5 にクランプ
    t = compute_sample_time_v2(short, TIME_PHASE_MID_P30)
    assert 50.0 <= t <= 60.0
    assert t == pytest.approx(short.end_sec - 0.5)


def test_compute_sample_time_v2_unknown_phase_raises() -> None:
    """未知 phase では ValueError。"""
    m = _meta()
    with pytest.raises(ValueError):
        compute_sample_time_v2(m, "unknown_phase")


def test_phase_definitions_anchor_values_are_known() -> None:
    """PHASE_DEFINITIONS の anchor 文字列は start/mid/end のいずれか。"""
    for phase, (anchor, _) in PHASE_DEFINITIONS.items():
        assert anchor in ("start", "mid", "end"), (
            f"{phase} の anchor={anchor} が想定外"
        )
