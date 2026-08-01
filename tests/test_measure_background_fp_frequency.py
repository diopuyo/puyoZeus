"""scripts.measure_background_fp_frequency の単体テスト (合成 board_log での検出ロジック確認)。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from scripts.measure_background_fp_frequency import (
    INDUCED_CHAIN_LOOKAHEAD_SEC,
    TRANSIENT_DURATION_SEC,
    SideFrame,
    check_induced_chain,
    distinct_stable_snapshots,
    find_background_fp_candidates,
    find_empty_to_color_cells,
    has_state_in_range,
    looks_like_board_log,
    load_side_frames_from_jsonl,
    measure_persistence,
)


def _empty_grid() -> np.ndarray:
    return np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int64)


def _stable(t_sec: float, grid: np.ndarray, frame_idx: int | None = None) -> SideFrame:
    return SideFrame(
        frame_idx=frame_idx if frame_idx is not None else int(t_sec * 30),
        t_sec=t_sec, state=BoardState.STABLE, grid=grid,
    )


def _non_stable(t_sec: float, state: BoardState) -> SideFrame:
    return SideFrame(frame_idx=int(t_sec * 30), t_sec=t_sec, state=state, grid=None)


class TestFindEmptyToColorCells:
    def test_detects_empty_to_color(self) -> None:
        prev = _empty_grid()
        curr = _empty_grid()
        curr[5, 2] = 1
        cells = find_empty_to_color_cells(prev, curr)
        assert cells == [(5, 2)]

    def test_ignores_color_to_color(self) -> None:
        prev = _empty_grid()
        prev[5, 2] = 3
        curr = _empty_grid()
        curr[5, 2] = 1
        assert find_empty_to_color_cells(prev, curr) == []

    def test_ignores_empty_to_unknown(self) -> None:
        prev = _empty_grid()
        curr = _empty_grid()
        curr[5, 2] = 10  # COLOR_UNKNOWN
        assert find_empty_to_color_cells(prev, curr) == []

    def test_ignores_hidden_row(self) -> None:
        prev = _empty_grid()
        curr = _empty_grid()
        curr[0, 2] = 1  # 隠し段 (row0)
        assert find_empty_to_color_cells(prev, curr) == []


class TestDistinctStableSnapshots:
    def test_dedups_identical_consecutive_grids(self) -> None:
        g = _empty_grid()
        frames = [_stable(0.0, g.copy()), _stable(0.1, g.copy()), _stable(0.2, g.copy())]
        snaps = distinct_stable_snapshots(frames)
        assert len(snaps) == 1

    def test_skips_non_stable(self) -> None:
        g1 = _empty_grid()
        g2 = _empty_grid()
        g2[3, 3] = 2
        frames = [_stable(0.0, g1), _non_stable(0.1, BoardState.TSUMO_FALL), _stable(0.2, g2)]
        snaps = distinct_stable_snapshots(frames)
        assert len(snaps) == 2


class TestBackgroundFpDetection:
    """でっち上げ候補検出 (設置/連鎖での正当な出現との弁別)。"""

    def _make_frames_with_gap(self, gap_state: BoardState | None) -> list[SideFrame]:
        prev_grid = _empty_grid()
        curr_grid = _empty_grid()
        curr_grid[5, 2] = 1  # 空->赤 (でっち上げ候補)
        frames = [_stable(0.0, prev_grid)]
        if gap_state is not None:
            frames.append(_non_stable(0.5, gap_state))
        frames.append(_stable(1.0, curr_grid))
        return frames

    def test_flags_candidate_when_no_explaining_event(self) -> None:
        frames = self._make_frames_with_gap(gap_state=None)
        sim = ChainSimulator()
        cands = find_background_fp_candidates(frames, "vX", "1P", "test", sim)
        assert len(cands) == 1
        assert (cands[0].row, cands[0].col) == (5, 2)
        assert cands[0].fabricated_color == 1

    def test_excludes_when_placement_event_between(self) -> None:
        frames = self._make_frames_with_gap(gap_state=BoardState.TSUMO_FALL)
        sim = ChainSimulator()
        cands = find_background_fp_candidates(frames, "vX", "1P", "test", sim)
        assert cands == []

    def test_excludes_when_ojama_fall_between(self) -> None:
        frames = self._make_frames_with_gap(gap_state=BoardState.OJAMA_FALL)
        sim = ChainSimulator()
        cands = find_background_fp_candidates(frames, "vX", "1P", "test", sim)
        assert cands == []

    def test_excludes_when_chain_refill_between(self) -> None:
        frames = self._make_frames_with_gap(gap_state=BoardState.CHAIN)
        sim = ChainSimulator()
        cands = find_background_fp_candidates(frames, "vX", "1P", "test", sim)
        assert cands == []

    def test_excludes_when_gravity_settle_between(self) -> None:
        frames = self._make_frames_with_gap(gap_state=BoardState.GRAVITY_SETTLE)
        sim = ChainSimulator()
        cands = find_background_fp_candidates(frames, "vX", "1P", "test", sim)
        assert cands == []


class TestHasStateInRange:
    def test_true_when_state_present(self) -> None:
        frames = [_stable(0.0, _empty_grid()), _non_stable(0.1, BoardState.CHAIN), _stable(0.2, _empty_grid())]
        assert has_state_in_range(frames, 0, 2, frozenset({BoardState.CHAIN})) is True

    def test_false_when_absent(self) -> None:
        frames = [_stable(0.0, _empty_grid()), _non_stable(0.1, BoardState.MENU), _stable(0.2, _empty_grid())]
        assert has_state_in_range(frames, 0, 2, frozenset({BoardState.CHAIN})) is False


class TestMeasurePersistence:
    def test_transient_resolves_quickly(self) -> None:
        g0 = _empty_grid()
        g1 = _empty_grid()
        g1[5, 2] = 1
        g2 = _empty_grid()  # 0.3秒後に自己修復 (空に戻る)
        snaps = [
            (0, _stable(0.0, g0)), (1, _stable(0.1, g1)), (2, _stable(0.4, g2)),
        ]
        dur, n_obs, resolved = measure_persistence(snaps, 1, 5, 2, 1)
        assert resolved is True
        assert dur == pytest.approx(0.3)
        assert dur <= TRANSIENT_DURATION_SEC

    def test_persists_to_end_of_log(self) -> None:
        g0 = _empty_grid()
        g1 = _empty_grid()
        g1[5, 2] = 1
        snaps = [(0, _stable(0.0, g0)), (1, _stable(2.0, g1))]
        dur, n_obs, resolved = measure_persistence(snaps, 1, 5, 2, 1)
        assert resolved is False
        assert n_obs == 1

    def test_persistent_beyond_threshold(self) -> None:
        g0 = _empty_grid()
        g1 = _empty_grid()
        g1[5, 2] = 1
        g2 = _empty_grid()  # 3秒後に解消 (閾値超え=持続的)
        snaps = [(0, _stable(0.0, g0)), (1, _stable(0.5, g1)), (2, _stable(3.5, g2))]
        dur, n_obs, resolved = measure_persistence(snaps, 1, 5, 2, 1)
        assert resolved is True
        assert dur > TRANSIENT_DURATION_SEC


class TestCheckInducedChain:
    def test_immediate_fire_detected_for_4plus_connection(self) -> None:
        grid = _empty_grid()
        grid[9, 0] = grid[9, 1] = grid[9, 2] = grid[9, 3] = 1  # 横4連結 (即消去対象)
        frame = _stable(1.0, grid)
        frames = [frame]
        sim = ChainSimulator()
        immediate, followed = check_induced_chain(frames, 0, frame, sim)
        assert immediate is True
        assert followed is False

    def test_no_immediate_fire_for_isolated_cell(self) -> None:
        grid = _empty_grid()
        grid[5, 2] = 1
        frame = _stable(1.0, grid)
        frames = [frame]
        sim = ChainSimulator()
        immediate, _followed = check_induced_chain(frames, 0, frame, sim)
        assert immediate is False

    def test_chain_followed_within_lookahead(self) -> None:
        grid = _empty_grid()
        grid[5, 2] = 1
        frame = _stable(1.0, grid)
        frames = [frame, _non_stable(1.0 + INDUCED_CHAIN_LOOKAHEAD_SEC - 0.1, BoardState.CHAIN)]
        sim = ChainSimulator()
        _immediate, followed = check_induced_chain(frames, 0, frame, sim)
        assert followed is True

    def test_chain_not_followed_beyond_lookahead(self) -> None:
        grid = _empty_grid()
        grid[5, 2] = 1
        frame = _stable(1.0, grid)
        frames = [frame, _non_stable(1.0 + INDUCED_CHAIN_LOOKAHEAD_SEC + 0.1, BoardState.CHAIN)]
        sim = ChainSimulator()
        _immediate, followed = check_induced_chain(frames, 0, frame, sim)
        assert followed is False


class TestJsonlLoader:
    def test_roundtrip_parses_state_and_grid(self, tmp_path: Path) -> None:
        grid = _empty_grid().tolist()
        grid2 = _empty_grid()
        grid2[5, 2] = 1
        row0 = {
            "frame_idx": 0, "t_sec": 0.0, "p1_state": "stable", "p2_state": "menu",
            "p1_confirmed": grid, "p2_confirmed": None,
        }
        row1 = {
            "frame_idx": 1, "t_sec": 0.1, "p1_state": "stable", "p2_state": "menu",
            "p1_confirmed": grid2.tolist(), "p2_confirmed": None,
        }
        path = tmp_path / "sample_boardlog.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in (row0, row1)) + "\n", encoding="utf-8")

        assert looks_like_board_log(path) is True
        out = load_side_frames_from_jsonl(path)
        assert len(out["1P"]) == 2
        assert out["1P"][0].state == BoardState.STABLE
        assert out["2P"][0].state == BoardState.MENU
        assert int(out["1P"][1].grid[5, 2]) == 1

    def test_looks_like_board_log_rejects_other_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "not_a_boardlog.jsonl"
        path.write_text(json.dumps({"route_id": "foo", "frame_idx": 0}) + "\n", encoding="utf-8")
        assert looks_like_board_log(path) is False

    def test_end_to_end_jsonl_detects_candidate(self, tmp_path: Path) -> None:
        """JSONLロード→検出まで通しで、正当イベント無しの空->色 出現がでっち上げ候補になる。"""
        prev_grid = _empty_grid().tolist()
        curr_grid = _empty_grid()
        curr_grid[5, 2] = 1
        rows = [
            {
                "frame_idx": 0, "t_sec": 0.0, "p1_state": "stable", "p2_state": "menu",
                "p1_confirmed": prev_grid, "p2_confirmed": None,
            },
            {
                "frame_idx": 1, "t_sec": 1.0, "p1_state": "stable", "p2_state": "menu",
                "p1_confirmed": curr_grid.tolist(), "p2_confirmed": None,
            },
        ]
        path = tmp_path / "e2e_boardlog.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        by_side = load_side_frames_from_jsonl(path)
        sim = ChainSimulator()
        cands = find_background_fp_candidates(by_side["1P"], "vTest", "1P", str(path), sim)
        assert len(cands) == 1
        assert (cands[0].row, cands[0].col) == (5, 2)
