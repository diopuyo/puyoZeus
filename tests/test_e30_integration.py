"""E30の予測層限定・死亡上限・既定OFFと因果的監査を検証する。"""
from __future__ import annotations
from copy import deepcopy
import inspect
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.chain import ChainSimulator
from src.exchange_death_inputs import maximum_response, hidden_proof
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_tracker import ExchangeChainRecord
from src.exchange_prefire_candidates import PrefireCandidates, placements, statistics
from scripts.replay_exchange_event_20260926 import replay
from scripts.audit_e30_predictions import CandidateTrace
from tests.test_e30_prefire_candidates import option


def prepare(options: list | None = None) -> tuple:
    """任意の候補を持つ予測器を作り、元予測と監査の境界を確認する。"""
    engine = PrefireCandidates(ChainSimulator())
    chain = ExchangeChainRecord('2P', 7, 1., 1.)
    entry = dict(chain=chain, options=options or [option(100), option(130)], elapsed=0.,
        original=(None, None, None), published=None, last=None,
        audit=dict(game=1, side='2P', chain_id=7, observations=[], initial=dict(candidates=2), used=False, uses=0))
    engine.entries[7] = entry
    engine._publish(entry)
    return engine, chain, entry


@pytest.mark.parametrize('call', [ExchangeEventOverlay, replay])
def test_default_off(call: object) -> None:
    assert inspect.signature(call).parameters['prefire_candidates'].default is False


def test_mean_send_is_mean_after_floor_not_floor_after_mean() -> None:
    engine, chain, entry = prepare([option(40), option(100)])
    assert chain.predicted_final_score == 70
    assert engine.sends([chain], 0) == [0, .5]
    assert entry['stats']['mean_send'] == .5


@pytest.mark.parametrize('count', [1, 2, 3])
def test_maximum_keeps_all_tied_boards(count: int) -> None:
    engine, chain, _ = prepare([option(400, color=c) for c in range(count)]+[option(100, color=4)])
    bound = engine.maximum(chain)
    assert bound['score'] == 400
    assert len(bound['options']) == count
    assert bound['prefire'] is True


@pytest.mark.parametrize('hidden_score,prefire_score,expected,boards',
    [(100, 200, 200, 1), (200, 100, 200, 1), (200, 200, 200, 2)])
def test_independent_predictions_use_more_optimistic_bound(
        hidden_score: int, prefire_score: int, expected: int, boards: int) -> None:
    hidden = dict(score=hidden_score, options=[option(hidden_score)], audit={})
    prefire = dict(score=prefire_score, options=[option(prefire_score, color=2)], audit={}, prefire=True)
    projection = NS(hidden_death=NS(maximum=lambda c: hidden), prefire=NS(maximum=lambda c: prefire))
    before = deepcopy((hidden, prefire))
    value = maximum_response(projection, object())
    assert value['score'] == expected and len(value['options']) == boards
    assert (hidden, prefire) == before


@pytest.mark.parametrize('outcomes,dead', [([True, True], True), ([True, False], False), ([False, True], False)])
def test_prefire_death_requires_all_maximum_boards(
        monkeypatch: pytest.MonkeyPatch, outcomes: list, dead: bool) -> None:
    from src import exchange_event_multilanding as module
    iterator = iter(outcomes)
    monkeypatch.setattr(module, 'cached_proof', lambda *a: dict(dead=next(iterator)))
    engine, chain, _ = prepare([option(400), option(400, color=2)])
    bound = engine.maximum(chain)
    result = hidden_proof(NS(), bound, (), 100, 1, 0., 3.)
    assert result['dead'] is dead
    assert result['reason'] == 'prefire_maximum_response'
    assert bound['audit']['uses'] == 1


def test_native_enumerates_all_22_orientations() -> None:
    values = placements(Board(), ((1, 2),), ChainSimulator())
    assert len(values) == 22
    assert not any(v.chain_result.chain_count for v in values)


def test_native_and_python_enumeration_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import puyo_core_bridge as native
    origin = Board()
    origin._grid[-1, :3] = 1
    sim = ChainSimulator(exclude_hidden_row_from_pop=True)
    fast = placements(origin, ((1, 2),), sim)
    monkeypatch.setattr(native, 'NATIVE_AVAILABLE', False)
    slow = placements(origin, ((1, 2),), sim)
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a.placed_board._grid, b.placed_board._grid)
        assert a.chain_result.chain_count == b.chain_result.chain_count


def test_audit_retains_retracted_predictions_and_unknown_final() -> None:
    engine, chain, entry = prepare()
    row = entry['audit']
    row['observations'] = [dict(candidates=2, mean_score=400), dict(candidates=0, mean_score=None)]
    engine.audit = [row]
    trace = CandidateTrace('test')
    trace.engine, trace.chains = engine, {(1, '2P', 7): chain}
    saved = deepcopy(entry['options'])
    assert trace.summary()['rows'][0]['first_error'] is None
    chain.score_delta, chain.score_ready_sec, chain.end_signal_sec = 300, 3., 3.
    result = trace.summary()['rows'][0]
    assert result['first_error'] == result['last_error'] == 100
    assert result['became_zero'] and not result['remaining']
    assert entry['options'] == saved


def test_existing_completion_is_not_enumerated() -> None:
    engine, chain, _ = prepare()
    engine.entries.clear()
    chain.predicted_chain_count = 2
    engine.seed(chain, None, [], 1, 0.)
    assert not engine.entries and not engine.audit


@pytest.mark.parametrize('unknown', [False, True])
def test_missing_origin_or_palette_does_not_invent_prediction(unknown: bool) -> None:
    engine = PrefireCandidates(ChainSimulator())
    chain = ExchangeChainRecord('1P', 1, 1., 1.)
    origin = Board() if unknown else None
    if unknown:
        origin._grid[0, 0] = 10
    engine.seed(chain, NS(before_board=origin), [], 1, 0.)
    assert chain.predicted_final_score is None
    assert engine.audit[0]['skipped'] in ('missing_origin', 'unknown_origin')


@pytest.mark.parametrize('source,label', [('S3_provisional', True), ('S3_landing', True),
    ('unavoidable_death', True), ('G_fe', False), ('confirmed_death', False), ('E16_current', False)])
def test_panel_marks_candidate_predictions_only(source: str, label: bool) -> None:
    from scripts.review_data_panel import _prediction_label
    engine, chain, _ = prepare()
    overlay = NS(tracker=NS(source=source), _prefire=engine, _midchain=None)
    assert bool(_prediction_label(overlay, NS(chains=[chain]), 1)) is label
