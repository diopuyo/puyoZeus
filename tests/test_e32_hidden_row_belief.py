"""E32の因果性・分布・候補の上限と予測境界を検証する。"""
from __future__ import annotations
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.chain import ChainSimulator
from src.board_state_machine import BoardState
from src.probabilistic_board import ProbabilisticCell
from src.hidden_row_belief import HiddenRowBelief, combinations, prior, MAX_COMBINATIONS

COLORS = (1, 2, 3, 4)


@pytest.mark.parametrize('color', COLORS)
def test_height_prior(color: int) -> None:
    grid = Board()._grid
    assert prior(grid, 0, COLORS).probs == {0: 1.}
    grid[1:, 0] = color
    assert prior(grid, 0, COLORS).probs == {0: .5, 1: .125, 2: .125, 3: .125, 4: .125}


def test_enumeration_mass_and_cap() -> None:
    cell = ProbabilisticCell.uniform((0, *COLORS))
    rows, mass = combinations([cell]*6)
    assert len(rows) == MAX_COMBINATIONS
    assert mass == pytest.approx(MAX_COMBINATIONS/5**6)
    assert sum(p for _, p in rows) == pytest.approx(1.)


def test_enumeration_stops_at_mass() -> None:
    rows, mass = combinations([ProbabilisticCell({0: .995, 1: .005})])
    assert rows == [((0,), 1.)] and mass == .995


@pytest.mark.parametrize('hidden_color', COLORS)
def test_reveal_scores_prior_before_update(hidden_color: int) -> None:
    engine = HiddenRowBelief(ChainSimulator(exclude_hidden_row_from_pop=True))
    grid = Board()._grid
    grid[1:, 0] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 3]
    grid[-1, 1:4] = 3
    engine.advance(grid, COLORS, 1.)
    before = dict(engine.cells[0].probs)
    board = Board.from_list(grid.tolist())
    board._grid[0, 0] = hidden_color
    after = engine.simulator.simulate(board).steps[0].board_after._grid
    engine.advance(after, COLORS, 2.)
    row = engine.audit[0]
    assert row['probs'] == before and row['actual'] == hidden_color
    assert row['prior_sec'] < row['observed_sec']
    assert row['log_loss'] == pytest.approx(-np.log(.125))
    assert engine.cells[0].probs == {0: 1.}


def test_next_and_visible_placement_reconstruct_hidden_color() -> None:
    engine = HiddenRowBelief(ChainSimulator())
    grid = Board()._grid
    grid[2:, 0] = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
    engine.advance(grid, COLORS, 1.)
    grid[1, 0] = 1
    engine.advance(grid, COLORS, 2., (1, 2))
    assert engine.cells[0].probs == {2: 1.}


def test_nonstable_confirmation_and_visible_unknown() -> None:
    engine = HiddenRowBelief(ChainSimulator())
    board = Board()
    assert not engine._usable(board, BoardState.GRAVITY_SETTLE)
    assert engine._usable(board, BoardState.GRAVITY_SETTLE)
    board._grid[-1, 0] = 10
    assert not engine._usable(board, BoardState.STABLE)


def test_raw_hidden_read_is_not_truth() -> None:
    engine = HiddenRowBelief(ChainSimulator())
    board = Board()
    board._grid[:, 0] = 1
    engine.advance(board._grid, COLORS, 1.)
    assert engine.cells[0].get(1) == .125


@pytest.mark.parametrize('col', range(6))
def test_visible_unknown_still_rejected(col: int) -> None:
    from src.exchange_hidden_row_belief import validate_visible
    board = Board()
    board._grid[-1, col] = 10
    assert validate_visible(board, dict(board=board._grid.tolist()), COLORS)[0] == 'unknown'


@pytest.mark.parametrize('col', range(6))
def test_hidden_unknown_allowed_and_origin_unchanged(col: int) -> None:
    from src.exchange_hidden_row_belief import validate_visible
    board = Board()
    board._grid[0, col] = 10
    reason, result = validate_visible(board, dict(board=board._grid.tolist()), COLORS)
    assert reason is None and result._grid[0, col] == 0
    assert board._grid[0, col] == 10


@pytest.mark.parametrize('weights,majority', [([.4,.3,.3],False), ([.5,.5],False), ([.51,.49],True)])
def test_final_board_requires_majority(weights: list, majority: bool) -> None:
    from src.exchange_hidden_row_belief import majority_statistics
    options = [dict(weight=w, board=[[i]], prefix=[40], score=40) for i,w in enumerate(weights)]
    assert (majority_statistics(options, 0)['board'] is not None) == majority


def test_posterior_filter_and_permanent_withdrawal() -> None:
    from src.exchange_hidden_row_belief import HiddenRowPrefire
    from src.exchange_event_tracker import ExchangeChainRecord
    engine = HiddenRowPrefire(ChainSimulator())
    chain = ExchangeChainRecord('1P', 1, 1., 1.)
    options = [dict(prefix=(40, score), score=score, board=Board()._grid.tolist(), weight=.5)
               for score in (360, 400)]
    entry = dict(options=options, last=(1,40), audit=dict(observations=[]), chain=chain,
                 original=(77., 1, None), elapsed=0.)
    engine._filter(entry, (2, 360), 2.)
    assert len(entry['options']) == 1 and entry['options'][0]['weight'] == 1.
    engine._publish(entry)
    assert chain.predicted_final_score == 360
    engine._filter(entry, (3, 700), 3.)
    engine._publish(entry)
    assert entry['audit']['withdrawn'] and chain.predicted_final_score == 77.


def test_game_reset_clears_only_beliefs() -> None:
    from src.exchange_hidden_row_belief import HiddenRowPrefire
    engine = HiddenRowPrefire(ChainSimulator())
    engine.calibration.append(dict(hit=True))
    engine.timelines[0].append((1., None))
    engine.reset()
    assert not engine.timelines[0] and engine.calibration == [dict(hit=True)]


def test_probability_average_is_not_mean_score(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import exchange_hidden_row_probability as probability
    from src.exchange_event_evaluator import ExchangeEndInput, FiringInput, StaticInput
    from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
    from src.exchange_event_tracker import ExchangeChainRecord
    chain = ExchangeChainRecord('1P', 1, 1., 1.)
    entry = dict(options=[dict(score=10, weight=.5), dict(score=90, weight=.5)])
    engine = NS(active=lambda c: entry)
    tracker = NS(current=NS(chains=[chain]), models=None)
    event = ExchangeEndInput(FiringInput(StaticInput(np.zeros(len(D_COLUMNS)), .5, 0),
        np.zeros((2,len(SIDE_COLUMNS))), (True,False)), np.zeros(2), np.array([50,0]), 0.)
    monkeypatch.setattr(probability, 'evaluate_exchange_event', lambda e,m: (e.scores_after[0]/100)**2)
    assert probability.weighted_s3(engine, tracker, event) == pytest.approx(.41)


def test_conditional_prediction_restored_on_error() -> None:
    from src.exchange_hidden_row_probability import conditional_scores
    chain = NS(predicted_final_score=50.)
    entry = dict(elapsed=0., stats=dict(mean_send=7.))
    with pytest.raises(RuntimeError):
        with conditional_scores(NS(active=lambda c: entry), [chain], (90,)):
            assert chain.predicted_final_score == 90
            raise RuntimeError('検証用')
    assert chain.predicted_final_score == 50 and entry['stats']['mean_send'] == 7.


@pytest.mark.parametrize('name', ['overlay','replay'])
def test_hidden_belief_default_off(name: str) -> None:
    import inspect
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    call = ExchangeEventOverlay if name == 'overlay' else replay
    assert inspect.signature(call).parameters['hidden_row_belief'].default is False


def test_belief_requires_snapshot() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    with pytest.raises(ValueError, match='prefire-snapshot'):
        ExchangeEventOverlay(None, None, None, hidden_row_belief=True)


def test_pair_disappears_into_two_hidden_cells() -> None:
    engine = HiddenRowBelief(ChainSimulator())
    board = Board()
    board._grid[1:, :2] = np.array([1,2,3,4]*3)[:, None]
    engine.advance(board._grid, COLORS, 1.)
    engine.queue, engine.pair = (1,2,3,4), (2,3)
    side = NS(state=BoardState.STABLE, confirmed_board=board, next_pair=(3,4), dnext_pair=(1,1))
    engine.observe(side, COLORS, 2.)
    assert engine.cells[0].probs == {2: .5, 3: .5}
    assert engine.cells[1].probs == {2: .5, 3: .5}


def test_palette_acquired_after_initial_observation() -> None:
    engine = HiddenRowBelief(ChainSimulator())
    board = Board()
    board._grid[1:, 0] = 1
    engine.advance(board._grid, (), 1.)
    engine.advance(board._grid, COLORS, 2.)
    assert engine.cells[0].get(1) == .125


def test_old_snapshot_does_not_use_new_belief() -> None:
    from src.exchange_hidden_row_belief import HiddenRowPrefire
    engine = HiddenRowPrefire(ChainSimulator())
    engine.evidence.counts.update(COLORS)
    old, future = HiddenRowBelief(engine.simulator), HiddenRowBelief(engine.simulator)
    board = Board()
    board._grid[1:, 0] = np.array([1,2,3,4]*3)
    old.advance(board._grid, COLORS, 1.)
    future.advance(board._grid, COLORS, 3.)
    old.cells[0], future.cells[0] = ProbabilisticCell.certain(1), ProbabilisticCell.certain(2)
    engine.timelines[0].extend([(1.,old),(3.,future)])
    _, _, cells = engine._options(NS(side='1P',trigger_sec=4.),board,dict(end_sec=2.))
    assert cells[0] == {1: 1.}


def test_history_copies_beliefs_without_copying_simulation_cache() -> None:
    from src.exchange_hidden_row_belief import HiddenRowPrefire
    engine = HiddenRowPrefire(ChainSimulator())
    side = NS(state=BoardState.STABLE, confirmed_board=Board(), next_pair=(1,2), dnext_pair=(3,4))
    engine.observe_history((side,side), 1., 1)
    saved = engine.timelines[0][-1][1]
    assert saved.simulator is engine.simulator
    assert saved.cells[0] is not engine.beliefs[0].cells[0]
    saved.cells[0].probs[1] = .5
    assert engine.beliefs[0].cells[0].probs == {0: 1.}


def test_landing_averages_candidate_probabilities_and_zero_incoming(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import exchange_hidden_row_probability as probability
    from src.exchange_event_evaluator import FiringInput, StaticInput
    from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
    from src.exchange_event_tracker import ExchangeChainRecord
    chain = ExchangeChainRecord('1P', 1, 1., 1.)
    chain.predicted_final_score = 40.
    entry = dict(options=[dict(score=0,weight=.5), dict(score=80,weight=.5)], elapsed=0., stats=dict(mean_send=.5))
    engine = NS(active=lambda c: entry)
    firing = FiringInput(StaticInput(np.zeros(len(D_COLUMNS)), .5, 0),
        np.zeros((2,len(SIDE_COLUMNS))), (True,False))
    tracker = NS(current=NS(chains=[chain]), models=None, firing=firing, _score_elapsed=0., hidden_row_belief=engine)
    projection = NS(death_record=None, _incoming=lambda *a: [0,int(chain.predicted_final_score>0)],
                    _probability_inputs=lambda *a: (.8,{}))
    snapshot = NS(total_dropped_to_p1=0, total_dropped_to_p2=0)
    monkeypatch.setattr(probability, 'evaluate_exchange_event', lambda e,m: e.scores_after[0]/100)
    value = probability.weighted_landing(projection,NS(tracker=tracker),snapshot,(),(),dict(p1=.5),2.)
    assert value[0] == pytest.approx(.4)
    assert chain.predicted_final_score == 40. and entry['stats']['mean_send'] == .5


def test_death_counterexample_keeps_all_surviving_boards() -> None:
    from src.exchange_hidden_row_belief import HiddenRowPrefire
    from src.exchange_event_tracker import ExchangeChainRecord
    engine = HiddenRowPrefire(ChainSimulator())
    chain = ExchangeChainRecord('1P', 1, 1., 1.)
    engine.entries[1] = dict(options=[dict(score=40,board=[[1]],weight=.1),
                                    dict(score=360,board=[[2]],weight=.9)], audit={})
    bound = engine.maximum(chain)
    assert bound['score'] == 360 and len(bound['options']) == 2
