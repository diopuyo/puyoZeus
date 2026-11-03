"""探索下限の上限流用と、得点履歴欠落の予告利用を防ぐ回帰検査。"""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.b3_column_spec import Mask
from src.b3_features import build_features, sl_group
from src.b3_score_ledger import ScoreLedgerV2, FrameObs, SideObs, snapshot_to_dict
from src.b3_search import IdealSearchResult, points_to_ojama
from src.b3_state import Episode, Fragment, SideState, GameState
from src.b3_source import load_lean
from src.b3_timing import TimingTable
from src.b3_timeaxis import _bound_material, _bound_q, _score_upper, time_axis_features
from src.scoring import ALL_CLEAR_BONUS, BASE_SCORE_PER_PUYO, MAX_BONUS_MULTIPLIER
from tests.b3_synth import build_record


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("active", [False, True])
def test_beam_result_never_reduces_upper_bound(full: bool, active: bool) -> None:
    search = IdealSearchResult((0.0,), (0,), 1, 1, False, full=full)
    q = _bound_q(72, 1, 0, 1, 0, 0, active, search)
    assert q == _bound_q(72, 1, 900000, 1, 0, 0, active)
    assert q >= points_to_ojama(_score_upper(74) + ALL_CLEAR_BONUS, 0)


def test_unknown_stage_uses_maximum_multiplier_without_double_material() -> None:
    q = _bound_q(8, 1, 0, 1, 50, 0, True)
    score = 10 * BASE_SCORE_PER_PUYO * MAX_BONUS_MULTIPLIER + 2 * ALL_CLEAR_BONUS
    assert q == 50 + points_to_ojama(score, 0) + 2
    assert q == _bound_q(8, 15, 0, 1, 50, 0, True)


def test_missing_origin_does_not_subtract_observed_erasures() -> None:
    side = SideState("1P")
    side.episode = Episode(0, 0, steps=[(1, 70, 10000)])
    assert _bound_material(side, (1, 2, 3, 4), 1)[0] == 78


def test_inconsistent_origin_keeps_full_board_material() -> None:
    side = SideState("1P")
    board = np.zeros((13, 6), dtype=np.int8)
    board[-1, :4] = 1
    side.episode = Episode(0, 0, frags=[Fragment("f", 0, board, None)], steps=[(1, 4, 40)])
    side.episode.prediction.reason = "observation_conflict"
    assert _bound_material(side, (1, 2, 3, 4), 1)[0] == 78


def test_prefire_material_includes_two_pairs_and_conflict_falls_back() -> None:
    side = SideState("1P")
    board = np.ones((13, 6), dtype=np.int8)
    side.episode = Episode(0, 0, frags=[Fragment("f", 0, board, {"board": board})],
                           steps=[(1, 4, 40)])
    assert _bound_material(side, (1, 2, 3, 4), 1)[0] == 78
    side.episode.steps = [(1, 90, 900)]
    assert _bound_material(side, (1, 2, 3, 4), 1)[0] == 78


def snapshot(ledger: ScoreLedgerV2, game: int, scores: tuple[int, int], start: int) -> dict:
    """二度の読みで開始得点を採用させる。"""
    for frame in range(start, start + 3):
        obs = FrameObs(t_sec=frame / 30, frame=frame, game_idx=game, match_active=True,
                       sides=tuple(SideObs(score=s) for s in scores))
        result = ledger.update(obs)
    return snapshot_to_dict(result)


@pytest.mark.parametrize("scores", [(2186, 0), (0, 2184), (2186, 2184)])
def test_incomplete_history_masks_both_sides_until_match_reset(scores: tuple[int, int]) -> None:
    ledger = ScoreLedgerV2()
    snap = snapshot(ledger, 1, scores, 0)
    row: dict = {}
    sl_group(row, SimpleNamespace(sl=ledger, sl_snap=snap))
    for side in ("p1", "p2"):
        assert row[f"mk_sl_{side}"] == Mask.HISTORY_INCOMPLETE
        assert np.isnan(row[f"sl_pending_{side}"])
        assert np.isnan(row[f"sl_pending_nf_{side}"])
        assert np.isnan(row[f"sl_board_excess_{side}"])
        assert np.isfinite(row[f"sl_generated_{side}"])
    later = snapshot(ledger, 1, scores, 30)
    assert later["history_incomplete"] and np.isnan(later["pending_p1"])
    clean = snapshot(ledger, 2, (0, 0), 60)
    assert not clean["history_incomplete"] and clean["pending_p1"] == 0


def test_pending_mask_reaches_time_axis(tmp_path: Path) -> None:
    gs = GameState(load_lean(build_record(tmp_path)))
    gs.t = gs.av_frame = 1.0
    for side in gs.sides:
        side.board = np.zeros((13, 6), dtype=np.int8)
        side.board_t = 1.0
    gs.sides[0].episode = Episode(1, 1, pred_l=1, pred_score=40)
    gs.sl_snap = snapshot(gs.sl.core, 1, (2186, 0), 0)
    row = build_features(gs, TimingTable())
    assert row['mk_ta_p1'] == Mask.HISTORY_INCOMPLETE
    assert np.isnan(row['ta_i_sl_p1']) and np.isnan(row['ta_m_sl_med_p1'])
    assert np.isfinite(row['ta_q_med_p1'])


def test_formula_before_score_adoption_keeps_erased_material_score() -> None:
    a, b = SideState('1P'), SideState('2P')
    a.episode = Episode(0, 0, pred_l=2, pred_score=1000)
    b.board = np.zeros((13, 6), dtype=np.int8)
    b.board[-1, :4] = 1
    b.episode = Episode(0, 0, frags=[Fragment('f', 0, b.board, None)], steps=[(0.5, 4, 30000)])
    ledger = SimpleNamespace(core=SimpleNamespace(chains=[], st=[SimpleNamespace(chain=None)] * 2))
    row = time_axis_features(a, b, 1, 1, (1, 2, 3, 4), TimingTable(), 1000,
                             sl_b_pending=1000, score_ledger=ledger)
    assert row['ta_q_med'] >= points_to_ojama(30000, 1 + row['ta_a_rem_sec'])
