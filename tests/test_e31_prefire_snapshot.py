"""E31の画像窓、整合棄却、予測専用性と撤回を検証する。"""
from __future__ import annotations
from copy import deepcopy
import inspect
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.chain import ChainSimulator
from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
from src.board_state_machine import BoardState
from src.prefire_snapshot_reader import validate_snapshot, vote_window, MIN_FRAMES
from src.exchange_prefire_snapshot import PrefireSnapshot
from src.exchange_event_tracker import ExchangeChainRecord
from src.exchange_event_overlay import ExchangeEventOverlay
from scripts.replay_exchange_event_20260926 import replay

COLORS = (1, 2, 3, 4)


def board() -> Board:
    """赤4個で一段目40点の小盤面。"""
    value = Board()
    value._grid[-1, :4] = 1
    return value


def sample(value: Board, stamp: float, quality: str = '') -> dict:
    return dict(cnn=value._grid.tolist(), hsv=value._grid.tolist(), t_sec=stamp, quality=quality)


def prepare(score: int = 40) -> tuple:
    engine = PrefireSnapshot(ChainSimulator())
    engine.evidence.counts.update(COLORS)
    origin = board()
    origin._grid[-1, 2:4] = 0
    engine.snapshots = {'1P': dict(board=board()._grid.tolist())}
    chain = ExchangeChainRecord('1P', 1, 2., 2.)
    chain.predicted_final_score = 7.
    event = NS(before_board=origin, mechanism=CHAIN_MECHANISM_FORMULA_READ,
               chain_count=1, total_score=score, trigger_sec=2.)
    engine.seed(chain, event, [], 1, 0.)
    result = NS(p1=NS(chain_event=event), p2=NS(chain_event=None))
    overlay = NS(tracker=NS(latest_chain=lambda label: chain if label == '1P' else None))
    return engine, chain, event, result, overlay


@pytest.mark.parametrize('call', [ExchangeEventOverlay, replay])
def test_default_off(call: object) -> None:
    assert inspect.signature(call).parameters['prefire_snapshot'].default is False


def test_pair_and_current_origin_not_mutated() -> None:
    origin = board()
    origin._grid[-1, 2:4] = 0
    before = origin._grid.copy()
    reason, accepted = validate_snapshot(origin, dict(board=board()._grid.tolist()), COLORS)
    assert reason is None and accepted is not None
    np.testing.assert_array_equal(origin._grid, before)


@pytest.mark.parametrize('change,reason', [('unknown','unknown'), ('palette','palette'),
    ('floating','floating'), ('many','origin_difference'), ('missing','missing_origin')])
def test_rejections(change: str, reason: str) -> None:
    value, origin = board(), board()
    if change == 'unknown':
        value._grid[0,0] = 10
    elif change == 'palette':
        value._grid[-1,0] = 5
    elif change == 'floating':
        value._grid[0,5] = 1
    elif change == 'many':
        origin = Board()
        value._grid[-1,:] = 1
    else:
        origin = None
    assert validate_snapshot(origin, dict(board=value._grid.tolist()), COLORS)[0] == reason


@pytest.mark.parametrize('missing', [None, dict(reason='no_landed_window',board=None)])
def test_missing_window(missing: dict | None) -> None:
    assert validate_snapshot(board(), missing, COLORS)[1] is None


def test_future_and_pre_window_frames_ignored() -> None:
    frames = [sample(board(), t) for t in (0., 0.1, 0.2, 2., 2.1, 2.2)]
    assert vote_window(frames, 2.)['reason'] == 'no_landed_window'


@pytest.mark.parametrize('reason', ['effect_glow','frame_diff=80','v_mean_delta=40'])
def test_bad_quality_not_voted(reason: str) -> None:
    frames = [sample(board(), 1+i*.03, reason) for i in range(MIN_FRAMES)]
    assert vote_window(frames, 2.)['board'] is None


def test_quality_gap_breaks_continuous_landing() -> None:
    frames = [sample(board(),1+i*.03,'effect_glow' if i==2 else '') for i in range(5)]
    assert vote_window(frames,2.)['board'] is None


def test_before_erasure_window_selected() -> None:
    frames = [sample(board(),1+i*.03) for i in range(3)]
    frames += [sample(Board(),1.2+i*.03) for i in range(3)]
    np.testing.assert_array_equal(vote_window(frames,2.)['board'], board()._grid)


def test_cell_majority_and_ties() -> None:
    other = board()
    other._grid[-1,0] = 2
    frames = [sample(board(),1), sample(other,1.03), sample(board(),1.06)]
    assert vote_window(frames,2.)['board'][-1][0] == 1
    frames.append(sample(other,1.09))
    assert vote_window(frames,2.)['board'][-1][0] == 10


def test_first_formula_mismatch_never_published() -> None:
    engine, chain, _, result, overlay = prepare(100)
    engine.observe(overlay,result,2.)
    assert chain.predicted_final_score == 7.
    assert engine.audit[0]['reason'] == 'first_score'
    assert not engine.provenance([chain])


def test_accept_restore_and_later_permanent_withdrawal() -> None:
    engine, chain, event, result, overlay = prepare()
    engine.observe(overlay,result,2.)
    assert chain.predicted_final_score == 40
    assert engine.provenance([chain])[0]['label'] == '発火前盤面から予測'
    engine.restore()
    assert chain.predicted_final_score == 7.
    chain.predicted_final_score = 9.
    event.chain_count, event.total_score = 2, 360
    engine.observe(overlay,result,3.)
    assert chain.predicted_final_score == 9.
    assert engine.audit[0]['withdrawn']
    event.chain_count, event.total_score = 1, 40
    engine.observe(overlay,result,4.)
    assert chain.predicted_final_score == 9. and not engine.active(chain)


def test_reset_keeps_audit_but_not_predictions() -> None:
    engine, chain, _, result, overlay = prepare()
    engine.observe(overlay,result,2.)
    engine.reset()
    assert not engine.entries and len(engine.audit) == 1


def test_accepted_snapshot_does_not_modify_stable_history() -> None:
    engine, chain, _, result, overlay = prepare()
    sides = tuple(NS(state=BoardState.STABLE, confirmed_board=board(), prefire_snapshot=None) for _ in range(2))
    saved = deepcopy([s.confirmed_board._grid for s in sides])
    engine.observe_colors(sides)
    engine.observe(overlay,result,2.)
    for previous, side in zip(saved,sides):
        np.testing.assert_array_equal(previous,side.confirmed_board._grid)


def test_final_score_is_never_selection_input() -> None:
    engine, chain, _, result, overlay = prepare()
    chain.score_delta = 123456
    engine.observe(overlay,result,2.)
    assert chain.predicted_final_score == 40


def test_completed_chain_has_no_prediction_label() -> None:
    engine, chain, _, result, overlay = prepare()
    engine.observe(overlay,result,2.)
    chain.end_signal_sec = chain.score_ready_sec = 3.
    chain.end_confirmed = True
    assert engine.active(chain) is None


@pytest.mark.parametrize('grid', [None, board()._grid.tolist()])
def test_snapshot_roundtrip_preserves_window_metadata(grid: list | None) -> None:
    from src.exchange_event_record import encode, decode
    snapshot = dict(reason='no_landed_window' if grid is None else None, board=grid)
    assert decode(encode(snapshot)) == snapshot


def test_board_roundtrip_keeps_existing_type() -> None:
    from src.exchange_event_record import encode, decode
    restored = decode(encode(board()))
    assert isinstance(restored, Board)
    np.testing.assert_array_equal(restored._grid, board()._grid)


def test_repeated_formula_does_not_change_revision() -> None:
    engine, chain, _, result, overlay = prepare()
    engine.observe(overlay,result,2.)
    revision = engine.revision
    engine.restore()
    engine.observe(overlay,result,2.1)
    assert engine.revision == revision
    assert chain.predicted_final_score == 40


def test_stage_gap_retracts_even_matching_later_score() -> None:
    engine, chain, event, result, overlay = prepare()
    engine.entries[1]['options'][0]['prefix'] = (40,360,1000)
    engine.observe(overlay,result,2.)
    engine.restore()
    event.chain_count, event.total_score = 3,1000
    engine.observe(overlay,result,3.)
    assert engine.audit[0]['withdraw_reason'] == 'stage_gap'
    assert chain.predicted_final_score == 7


@pytest.mark.parametrize('grid', [[], [[0]*6]*12, [[0]*5]*13])
def test_invalid_snapshot_shape_is_rejected(grid: list) -> None:
    assert validate_snapshot(board(), dict(board=grid), COLORS)[0] == 'invalid_shape'


def test_four_cell_correction_limit() -> None:
    origin = board()
    value = board()
    value._grid[-1,:4] = 2
    assert validate_snapshot(origin,dict(board=value._grid.tolist()),COLORS)[0] is None
    value._grid[-1,4] = 2
    assert validate_snapshot(origin,dict(board=value._grid.tolist()),COLORS)[0] == 'origin_difference'


def test_snapshot_panel_label_is_separate_from_candidate_label() -> None:
    from scripts.review_data_panel import _prediction_label
    engine, chain, _, result, overlay = prepare()
    engine.observe(overlay,result,2.)
    display = NS(tracker=NS(source='S3_provisional'), _prefire=engine, _midchain=None)
    assert _prediction_label(display, NS(chains=[chain]), 2.) == '（発火前盤面から予測）'


@pytest.mark.parametrize('next_game', [1,2])
def test_live_window_matches_recorded_window_and_resets(next_game: int) -> None:
    from src.prefire_snapshot_reader import PrefireSnapshotReader
    reader = PrefireSnapshotReader(reader=object())
    reader.read_frame = lambda frame, idx, stamp: sample(board(),stamp)
    sides = (NS(chain_event=None),NS(chain_event=None))
    dummy = np.zeros((2,2,3),dtype=np.uint8)
    frames = [sample(board(),t) for t in (1.,1.03,1.06)]
    for frame in frames:
        reader.update(dummy,sides,frame['t_sec'],1)
    sides[0].chain_event = NS(trigger_sec=1.1)
    result = reader.update(dummy,sides,1.1,next_game)[0]
    if next_game == 1:
        assert result == vote_window(frames,1.1)
        assert result['end_sec'] < 1.1
    else:
        assert result['reason'] == 'no_landed_window'
