"""満杯盤面ラベルシート準備 (scripts/build_full_board_label_sheet.py) の単体テスト。

実npz/実動画は使わず、合成データのみで候補選定ロジック
(セル数閾値・動画上限・位相配分) を検証する。
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.build_full_board_label_sheet import (
    GameCandidate,
    PHASE_EARLY,
    PHASE_LATE,
    PHASE_MID,
    PRIMARY_MIN_OCCUPANCY,
    SECONDARY_MIN_OCCUPANCY,
    TIER_PRIMARY,
    TIER_SECONDARY,
    build_game_time_bounds,
    classify_phase,
    classify_tier,
    compute_occupancy,
    encode_grid_string,
    select_candidates,
)
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_RED, COLOR_UNKNOWN


def _make_grid(n_filled: int, fill_color: int = COLOR_RED) -> np.ndarray:
    """非空セル数 n_filled の合成グリッドを作る (盤面下段から埋める)。"""
    grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
    flat = grid.reshape(-1)
    flat[-n_filled:] = fill_color if n_filled > 0 else COLOR_EMPTY
    return flat.reshape(BOARD_ROWS, BOARD_COLS)


def _make_candidate(
    *, video_id: str = "video_c1", side: str = "1P", game_idx: int = 0,
    t_sec: float = 10.0, occupancy: int = 60, phase: str = PHASE_LATE,
    tier: str = TIER_PRIMARY, frac: float = 0.9,
) -> GameCandidate:
    return GameCandidate(
        video_id=video_id, side=side, game_idx=game_idx, frame_idx=0, t_sec=t_sec,
        grid=_make_grid(occupancy), occupancy=occupancy, tier=tier, phase=phase,
        game_progress_frac=frac,
    )


# =============================================================================
# compute_occupancy / classify_tier
# =============================================================================

class TestComputeOccupancy:
    def test_empty_grid_is_zero(self) -> None:
        assert compute_occupancy(_make_grid(0)) == 0

    def test_partial_fill_counts_correctly(self) -> None:
        assert compute_occupancy(_make_grid(37)) == 37

    def test_full_board_is_total_cells(self) -> None:
        assert compute_occupancy(_make_grid(BOARD_ROWS * BOARD_COLS)) == BOARD_ROWS * BOARD_COLS


class TestClassifyTier:
    def test_below_secondary_threshold_is_none(self) -> None:
        assert classify_tier(SECONDARY_MIN_OCCUPANCY - 1) is None

    def test_secondary_band(self) -> None:
        assert classify_tier(SECONDARY_MIN_OCCUPANCY) == TIER_SECONDARY
        assert classify_tier(PRIMARY_MIN_OCCUPANCY - 1) == TIER_SECONDARY

    def test_primary_band(self) -> None:
        assert classify_tier(PRIMARY_MIN_OCCUPANCY) == TIER_PRIMARY
        assert classify_tier(78) == TIER_PRIMARY


# =============================================================================
# build_game_time_bounds / classify_phase
# =============================================================================

class TestBuildGameTimeBounds:
    def test_bounds_span_both_sides(self) -> None:
        video_ids = np.array(["video_c1", "video_c1", "video_c1"])
        game_idxs = np.array([0, 0, 0])
        t_secs = np.array([1.0, 5.0, 3.0])
        bounds = build_game_time_bounds(video_ids, game_idxs, t_secs)
        assert bounds[("video_c1", 0)] == (1.0, 5.0)

    def test_separate_games_are_independent(self) -> None:
        video_ids = np.array(["video_c1", "video_c1"])
        game_idxs = np.array([0, 1])
        t_secs = np.array([10.0, 999.0])
        bounds = build_game_time_bounds(video_ids, game_idxs, t_secs)
        assert bounds[("video_c1", 0)] == (10.0, 10.0)
        assert bounds[("video_c1", 1)] == (999.0, 999.0)


class TestClassifyPhase:
    def test_single_snapshot_game_is_late(self) -> None:
        phase, frac = classify_phase(5.0, 5.0, 5.0)
        assert phase == PHASE_LATE
        assert frac == 1.0

    def test_early_fraction(self) -> None:
        phase, frac = classify_phase(t_sec=1.0, lo=0.0, hi=100.0)
        assert phase == PHASE_EARLY
        assert frac == pytest.approx(0.01)

    def test_mid_fraction(self) -> None:
        phase, _frac = classify_phase(t_sec=50.0, lo=0.0, hi=100.0)
        assert phase == PHASE_MID

    def test_late_fraction(self) -> None:
        phase, _frac = classify_phase(t_sec=90.0, lo=0.0, hi=100.0)
        assert phase == PHASE_LATE

    def test_boundary_values_are_inclusive_of_higher_band(self) -> None:
        # frac=0.66 ちょうど -> 終盤、frac=0.33 ちょうど -> 中盤
        phase_late, _ = classify_phase(t_sec=66.0, lo=0.0, hi=100.0)
        phase_mid, _ = classify_phase(t_sec=33.0, lo=0.0, hi=100.0)
        assert phase_late == PHASE_LATE
        assert phase_mid == PHASE_MID


# =============================================================================
# encode_grid_string
# =============================================================================

class TestEncodeGridString:
    def test_encodes_rows_separated_by_slash(self) -> None:
        grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
        encoded = encode_grid_string(grid)
        rows = encoded.split("/")
        assert len(rows) == BOARD_ROWS
        assert all(r == "0" * BOARD_COLS for r in rows)

    def test_unknown_color_encoded_as_u(self) -> None:
        grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
        grid[1, 2] = COLOR_UNKNOWN
        encoded = encode_grid_string(grid)
        row1 = encoded.split("/")[1]
        assert row1[2] == "U"

    def test_ojama_distinguished_from_unknown(self) -> None:
        grid = np.full((BOARD_ROWS, BOARD_COLS), COLOR_EMPTY, dtype=np.int64)
        grid[0, 0] = COLOR_OJAMA
        grid[0, 1] = COLOR_UNKNOWN
        row0 = encode_grid_string(grid).split("/")[0]
        assert row0[0] == "9"
        assert row0[1] == "U"
        assert row0[0] != row0[1]


# =============================================================================
# select_candidates (動画上限・位相配分)
# =============================================================================

class TestSelectCandidates:
    def test_respects_max_per_video_cap(self) -> None:
        pool = [
            _make_candidate(video_id="video_c1", occupancy=60 + i, t_sec=float(i))
            for i in range(5)
        ]
        selected = select_candidates(pool, target_total=10, max_per_video=2)
        assert len(selected) == 2

    def test_spreads_across_videos_before_taking_second_from_one(self) -> None:
        pool = []
        for v in range(5):
            for i in range(2):
                pool.append(_make_candidate(
                    video_id=f"video_c{v}", occupancy=60 + i, t_sec=float(i),
                ))
        selected = select_candidates(pool, target_total=5, max_per_video=2)
        by_video = {}
        for c in selected:
            by_video[c.video_stem] = by_video.get(c.video_stem, 0) + 1
        # 5件を5動画に配ろうとするので、1動画から2件取られるのは1本のみのはず
        assert max(by_video.values()) <= 2
        assert len(by_video) >= 3  # 1動画に偏らず複数動画に広がる

    def test_prefers_primary_tier_over_secondary(self) -> None:
        pool = [
            _make_candidate(video_id="video_c1", occupancy=56, tier=TIER_SECONDARY, t_sec=1.0),
            _make_candidate(video_id="video_c2", occupancy=70, tier=TIER_PRIMARY, t_sec=1.0),
        ]
        selected = select_candidates(pool, target_total=1, max_per_video=2)
        assert len(selected) == 1
        assert selected[0].video_stem == "c2"

    def test_mixes_mid_phase_per_target_fraction(self) -> None:
        late_pool = [
            _make_candidate(video_id=f"video_late{i}", phase=PHASE_LATE, t_sec=1.0)
            for i in range(20)
        ]
        mid_pool = [
            _make_candidate(video_id=f"video_mid{i}", phase=PHASE_MID, t_sec=1.0)
            for i in range(20)
        ]
        selected = select_candidates(
            late_pool + mid_pool, target_total=10, max_per_video=2,
            mid_phase_target_fraction=0.2,
        )
        n_mid = sum(1 for c in selected if c.phase == PHASE_MID)
        assert n_mid == 2  # round(10*0.2)

    def test_degrades_gracefully_when_mid_pool_insufficient(self) -> None:
        # 中盤候補が1件しかない -> 中盤枠を埋められないが処理は継続し終盤で埋める
        late_pool = [
            _make_candidate(video_id=f"video_late{i}", phase=PHASE_LATE, t_sec=1.0)
            for i in range(10)
        ]
        mid_pool = [_make_candidate(video_id="video_mid0", phase=PHASE_MID, t_sec=1.0)]
        selected = select_candidates(
            late_pool + mid_pool, target_total=5, max_per_video=2,
            mid_phase_target_fraction=0.2,
        )
        assert len(selected) == 5
        assert sum(1 for c in selected if c.phase == PHASE_MID) == 1

    def test_empty_pool_returns_empty(self) -> None:
        assert select_candidates([], target_total=10) == []
