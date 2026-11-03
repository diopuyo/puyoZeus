"""試合外除去・STABLE窒息監査・欠測mask・観測段下限の回帰検査。"""
from pathlib import Path

import numpy as np
import pytest

from src.b3_column_spec import COLUMN_SPECS, Mask
from src.b3_features import build_features, chain_group
from src.b3_rows import _track_stable_death, generate_rows, label_vs_death_audit
from src.b3_source import EXCLUDED_VIDEOS, MATCH_END_GRACE_SEC, load_lean, winners_by_overlap
from src.b3_state import GameState, SideState
from src.b3_timing import TimingTable
from tests.b3_synth import build_record


def test_ranges_use_overlap_not_game_id(tmp_path: Path) -> None:
    lean = load_lean(build_record(tmp_path))
    start, end = float(lean.t_sec.min()), float(lean.t_sec.max())
    winners, mapping = winners_by_overlap(lean, {44: (start - 10, end + 100), 49: (end + 101, end + 200)})
    assert winners[44] is not None and mapping[44]["lean_game"] == 1
    assert mapping[44]["start_sec"] == start
    assert mapping[44]["end_sec"] == end + MATCH_END_GRACE_SEC
    assert winners[49] is None and mapping[49]["start_sec"] is None


def test_outside_frames_excluded_before_state_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = build_record(tmp_path)
    import src.b3_rows as module
    monkeypatch.setattr(module, "_make_row", lambda gs, *args: {"t_sec": gs.t, "av_max": gs.t})
    all_rows, _ = generate_rows(rec, "synth", TimingTable(), winners={1: 1},
                                match_mapping={1: {"start_sec": 0, "end_sec": 1000}})
    span = {1: {"start_sec": 5, "end_sec": 10}}
    rows, info = generate_rows(rec, "synth", TimingTable(), winners={1: 1}, match_mapping=span)
    assert rows and all(5 <= r["t_sec"] <= 10 for r in rows)
    assert rows[0]["t_sec"] == 5
    assert len(all_rows) > len(rows)
    assert info["excluded_outside_match_frames"] == info["frames"] - info["frames_in_match"] > 0
    assert info["excluded_outside_match_rows"] == 0  # 試合外は行を作る前に遮断する。


def _death_frame(frame: int, states: tuple[str, str], dead: tuple[bool, bool]) -> dict:
    sides = []
    for state, occupied in zip(states, dead):
        board = np.zeros((13, 6), dtype=np.int8)
        board[1, 2] = int(occupied)
        sides.append({"state": state, "confirmed_board": board.tolist()})
    return {"game_idx": 44, "frame": frame, "t_sec": float(frame), "sides": sides}


@pytest.mark.parametrize("dead,winner,judged,bad", [((False, True), 1, 1, 0),
                                                   ((True, False), 1, 1, 1),
                                                   ((True, True), 1, 0, 0),
                                                   ((False, False), 0, 0, 0)])
def test_death_audit_uses_stable_not_last_row(dead: tuple[bool, bool], winner: int, judged: int, bad: int) -> None:
    snapshots: dict = {}
    _track_stable_death(snapshots, _death_frame(1, ("STABLE", "STABLE"), dead))
    _track_stable_death(snapshots, _death_frame(2, ("TSUMO_FALL", "TSUMO_FALL"), (True, False)))
    rows = [{"game_idx": 44, "frame_idx": 3, "won_p1": winner, "is_dead_p1": 1, "is_dead_p2": 0}]
    audit = label_vs_death_audit(rows, snapshots)
    assert (audit["judged"], audit["mismatch"]) == (judged, bad)
    assert audit["undetermined"] == 1 - judged


def test_chain_invalidates_pre_chain_dead_board() -> None:
    snapshots: dict = {}
    _track_stable_death(snapshots, _death_frame(1, ("STABLE", "STABLE"), (True, False)))
    _track_stable_death(snapshots, _death_frame(2, ("CHAIN", "STABLE"), (True, True)))
    rows = [{"game_idx": 44, "frame_idx": 2, "won_p1": 1}]
    assert label_vs_death_audit(rows, snapshots)["judged"] == 0
    _track_stable_death(snapshots, _death_frame(3, ("STABLE", "STABLE"), (False, True)))
    assert label_vs_death_audit(rows, snapshots)["judged"] == 1
    assert label_vs_death_audit(rows, snapshots)["mismatch"] == 0
    assert label_vs_death_audit(rows)["judged"] == 0


def test_backfilled_steps_without_coherent_candidate_are_unpredictable() -> None:
    st = SideState("1P")
    for t in (1.0, 1.01):
        st.on_formula({"t_sec": t, "observed_at_sec": t, "erased": 4, "product": 40})
    st.on_fire_start({"fire_id": "a", "trigger_sec": 1.0, "observed_at_sec": 1.02,
                      "chain_before": np.zeros((13, 6), dtype=np.int8).tolist()})
    row: dict = {}
    chain_group(row, st, "p1", 1.02, TimingTable(), 0)
    assert row["chain_n_steps_p1"] == 2
    assert np.isnan(row["chain_pred_l_p1"])
    assert np.isnan(row["chain_pred_score_p1"])
    assert row["chain_observed_score_p1"] == 80
    assert row["chain_pred_final_p1"] is None
    assert row["mk_chain_p1"] == Mask.UNPREDICTABLE
    st.on_formula({"t_sec": 1.03, "observed_at_sec": 1.03, "erased": 4, "product": 40})
    assert st.episode.prediction.observed_count == 3
    assert np.isnan(st.episode.pred_l)


def test_missing_icon_masks_all_affected_time_axis_columns(tmp_path: Path) -> None:
    gs = GameState(load_lean(build_record(tmp_path)))
    gs.t = gs.av_frame = 1.0
    for st in gs.sides:
        st.board = np.zeros((13, 6), dtype=np.int8)
        st.board_t = 1.0
    gs.sides[0].on_fire_start({"fire_id": "a", "trigger_sec": 1.0, "observed_at_sec": 1.0,
                               "chain_before": gs.sides[0].board.tolist()})
    # この検査は連鎖予測が有効な条件で、アイコン欠測だけを対象にする。
    gs.sides[0].episode.pred_l = 1
    gs.sides[0].episode.pred_score = 40
    # 台帳が利用可能な条件も再現し、アイコン欠測だけを検出する。
    from src.b3_score_ledger_port import validate_snapshot
    frame = _death_frame(1, ("STABLE", "STABLE"), (False, False))
    frame.update(available_at_sec=1.0, match_active=True)
    for raw in frame["sides"]:
        raw.update(score=0, icon=None)
    gs.sl.update(dict(frame, frame=0, t_sec=0.97), ())
    gs.sl_snap = validate_snapshot(gs.sl.update(frame, ()))
    row = build_features(gs, TimingTable())
    assert row["mk_sl_p1"] == Mask.OK
    assert np.isnan(row["ta_i_icon_p1"])
    assert row["mk_ta_p1"] == Mask.NOT_YET_AVAILABLE
    assert all(row[c.mask] != Mask.OK for c in COLUMN_SPECS if c.mask and np.isnan(row[c.name]))


def test_completed_videos_no_longer_excluded() -> None:
    assert not EXCLUDED_VIDEOS
