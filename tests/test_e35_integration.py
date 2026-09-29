"""E35の全候補・既定OFF・表示と既存探索への接続を検証する。"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_post_counter_bound import PostCounterDeathBound
from src.exchange_event_overlay import ExchangeEventOverlay
from scripts.review_data_panel import _prediction_label


def setup() -> tuple:
    """既知4色・完走予測ありの独立した死亡入力を作る。"""
    engine = PostCounterDeathBound()
    for evidence in engine.colors:
        evidence.counts.update({1: 10, 2: 10, 3: 10, 4: 10})
    chain = NS(chain_id=2, predicted_final_board=Board()._grid.tolist(), predicted_final_score=100)
    overlay = NS(_game=1, tracker=NS(latest_chain=lambda side: chain, _score_elapsed=0))
    projection = NS(_known_budget=lambda *a: True, _chaining=lambda *a: True)
    context = dict(hidden=[dict(score=100, options=[dict(board=Board()._grid.tolist())]*2), None],
                   certain=[True, True], verified=[True, True], credit=[0, 0],
                   replies=[Board(), Board()], incoming=[171, 0])
    return engine, projection, overlay, context


@pytest.mark.parametrize('results', ([True, True], [False, True], [True, False]))
def test_every_candidate_must_die(monkeypatch: pytest.MonkeyPatch, results: list[bool]) -> None:
    import src.exchange_post_counter_bound as module
    engine, projection, overlay, context = setup()
    calls = []
    def prove(*args: object) -> dict:
        calls.append(args)
        return dict(dead=results[len(calls)-1])
    monkeypatch.setattr(module, 'prove_post_counter', prove)
    value = engine.prove(projection, overlay, 0, (), 171, 1, context, 2.)
    assert value['dead'] == all(results) and len(calls) == 2


def test_live_next_is_used_instead_of_prefire_history(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.exchange_post_counter_bound as module
    engine, projection, overlay, context = setup()
    engine.queues = ((3, 4, 1, 2), ())
    queues = []
    def prove(board: Board, queue: tuple, *args: object) -> dict:
        queues.append(queue)
        return dict(dead=False)
    monkeypatch.setattr(module, 'prove_post_counter', prove)
    engine.prove(projection, overlay, 0, (NS(queue=(1, 1, 2, 2)),), 171, 1, context, 2.)
    assert queues == [(3, 4, 1, 2)]*2


def test_no_proof_keeps_original_death_list(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, projection, overlay, context = setup()
    monkeypatch.setattr(engine, 'prove', lambda *a: dict(dead=False, reason='bound_limit'))
    value = dict(dead_sides=[])
    engine.evaluate(projection, overlay, (), (1, 1), 2., value, context)
    assert value['dead_sides'] == []


def test_new_engine_is_off_by_default() -> None:
    overlay = ExchangeEventOverlay(NS(), lambda *a: None, lambda *a: None)
    assert overlay._landing_projection.post_counter_bound is None


def test_flag_requires_production_death_contract() -> None:
    with pytest.raises(ValueError, match='E35'):
        ExchangeEventOverlay(NS(), lambda *a: None, lambda *a: None, post_counter_death_bound=True)


def test_prediction_label_is_explicit() -> None:
    overlay = NS(tracker=NS(source='unavoidable_death'),
                 _landing_projection=NS(death=dict(post_counter_bound=[dict(dead=True)])))
    assert _prediction_label(overlay, NS(), 1) == '（打ち返し後の負け確定・予測込み）'
    overlay.tracker.source = 'confirmed_death'
    assert _prediction_label(overlay, NS(), 1) == ''
    overlay.tracker.source = 'S3_landing'
    assert _prediction_label(overlay, NS(), 1) == ''


def test_boundary_resets_color_evidence() -> None:
    engine, *_ = setup()
    side = NS(state=BoardState.CHAIN, confirmed_board=None, next_pair=(1, 2), dnext_pair=(3, 4))
    engine.observe(NS(p1=side, p2=side), 2)
    assert engine.colors[0].active() == ()
    assert engine.queues[0] == (1, 2, 3, 4)
