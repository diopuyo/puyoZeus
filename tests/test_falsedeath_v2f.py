"""E35‴の正負対照・欠測・候補同定・既定OFF互換を検証する。"""
from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest

from src.board import Board
from src.exchange_post_counter_bound import PostCounterDeathBound


def scenario(enabled: bool = True) -> tuple[Any, ...]:
    """自連鎖後にも受け切れない盤面と、実読された1段目を用意する。"""
    board = Board()
    board._grid[-8:] = 9
    option = dict(board=board._grid.tolist(), prefix=(40,))
    chain = NS(chain_id=1, trigger_sec=0., end_signal_sec=None,
        predicted_final_score=40, predicted_chain_count=1, predicted_final_board=board._grid.tolist(),
        score_before=1000, display_score=1040, formula_total=40, score_delta=40, drop_bonus_score=0)
    engine = PostCounterDeathBound(current_puyo_integrity_v2c=True, completion_prefix_integrity=enabled)
    engine.states = ('CHAIN', 'CHAIN')
    engine.queues = ((1, 2, 3, 4),)*2
    engine.colors = [NS(active=lambda: (1, 2, 3, 4))]*2
    engine.formula_observations[(0, 1)] = (1, 40)
    signal = NS(current_puyo_observation=lambda q: dict(firing_placed=True, next_unchanged=True, ended=False))
    overlay = NS(_game=0, _signals={1: signal},
        tracker=NS(latest_chain=lambda side: chain, _score_elapsed=0.))
    projection = NS(_chaining=lambda *a: True, _known_budget=lambda *a: True,
        safety=NS(recovery=NS(entries={1: dict(options=[option])})))
    overlay._landing_projection = projection
    context = dict(hidden=[None]*2, certain=[True]*2, credit=[0]*2, verified=[True]*2, replies=[board]*2)
    return engine, projection, overlay, chain, context


def prove(case: tuple[Any, ...]) -> dict:
    """実E35探索へ同じ入力を渡す。"""
    engine, projection, overlay, _, context = case
    return engine.prove(projection, overlay, 0, (), 171, 1, context, 1.)


def test_matching_self_chain_proves_death() -> None:
    case = scenario()
    result = prove(case)
    assert result['dead']
    assert result['completion_prefix_check']['accepted']
    assert case[0].audit[-1]['completion_prefix_check']['score_display_increment'] == 40


def test_mia7_third_formula_rejects_first_step_prediction() -> None:
    case = scenario()
    case[0].formula_observations[(0, 1)] = (3, 1180)
    case[3].formula_total = 1180
    case[3].display_score = 2180
    assert not prove(case)['dead']
    check = case[0].audit[-1]['completion_prefix_check']
    assert check['reason'] == 'prefix_mismatch'
    assert check['observed_count'] == 3 and check['predicted'][0]['counts'] == [1]


@pytest.mark.parametrize('missing', ['formula', 'formula_total', 'display_score', 'score_before', 'candidate'])
def test_unobserved_does_not_prove(missing: str) -> None:
    case = scenario()
    if missing == 'formula':
        case[0].formula_observations.clear()
    elif missing == 'candidate':
        case[1].safety.recovery.entries.clear()
    else:
        setattr(case[3], missing, None)
    assert not prove(case)['dead']
    assert case[0].audit[-1]['completion_prefix_check']['reason'] == 'prefix_unobserved'


@pytest.mark.parametrize('field', ['display_score', 'formula_total'])
def test_independent_score_mismatch(field: str) -> None:
    case = scenario()
    setattr(case[3], field, getattr(case[3], field)+1)
    assert not prove(case)['dead']


def test_default_off_does_not_add_audit_or_reject() -> None:
    assert not PostCounterDeathBound().completion_prefix_integrity
    case = scenario(False)
    case[0].formula_observations.clear()
    assert prove(case)['dead']
    assert 'completion_prefix_check' not in case[0].audit[-1]


def test_revision_changes_when_score_changes() -> None:
    engine, projection, overlay, chain, _ = scenario()
    engine.refresh_current_observation(overlay)
    revision = engine.revision
    engine.refresh_current_observation(overlay)
    assert engine.revision == revision
    chain.display_score += 1
    engine.refresh_current_observation(overlay)
    assert engine.revision == revision+1


def test_all_selected_hidden_candidates_must_match() -> None:
    case = scenario()
    board = case[3].predicted_final_board
    case[4]['hidden'][0] = dict(score=40, options=[dict(board=board, prefix=(40,)),
                                                  dict(board=board, prefix=(50,))])
    assert not prove(case)['dead']


def test_v2f_only_adds_preregistered_flag() -> None:
    from scripts.falsedeath_replay_20261001 import enabled_flags
    assert enabled_flags('v2f') == enabled_flags('v2e') + ['--post-counter-completion-prefix-integrity']


@pytest.mark.parametrize('mechanism,trigger', [('formula', 0.), ('formula_read', -1.)])
def test_estimated_or_previous_chain_formula_is_not_observation(mechanism: str, trigger: float) -> None:
    from src.exchange_post_counter_prefix import observe_formula
    case = scenario()
    observed = {}
    side = NS(chain_event=NS(mechanism=mechanism, trigger_sec=trigger, chain_count=1, total_score=40))
    observe_formula(case[2], NS(p1=side, p2=NS(chain_event=None)), observed)
    assert not observed


def test_drop_bonus_is_separated_from_chain_score() -> None:
    case = scenario()
    case[3].display_score += 2
    case[3].drop_bonus_score = 2
    assert prove(case)['dead']


def test_stale_completion_is_not_used() -> None:
    case = scenario()
    case[3].predicted_final_score += 1
    assert not prove(case)['dead']


def test_original_prediction_survives_recovery_candidate_deletion() -> None:
    case = scenario()
    case[0].base_predictions[1] = case[1].safety.recovery.entries[1]['options'][0]
    case[1].post_counter_bound = case[0]
    case[1].safety.recovery.entries[1]['options'] = []
    assert prove(case)['dead']


def test_absolute_midchain_number_is_preserved() -> None:
    case = scenario()
    case[0].formula_observations[(0, 1)] = (3, 1180)
    case[3].formula_total = 1180
    case[3].display_score = 2180
    case[4]['hidden'][0] = dict(score=1180, options=[
        dict(board=case[3].predicted_final_board, prefix={3: 1180})])
    assert prove(case)['dead']


def test_default_off_on_all_entry_points() -> None:
    import inspect
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['post_counter_completion_prefix_integrity'].default is False


def test_block_classification_uses_same_game_side_and_future() -> None:
    from scripts.falsedeath_v2f_blocks import classify, counts
    row = dict(game=1, side='2P', chain_id=3, t_sec=10.,
               completion_prefix_check=dict(accepted=False, reason='prefix_unobserved'))
    terminals = [dict(game=1, side='2P', stamp=9.), dict(game=2, side='2P', stamp=11.),
                 dict(game=1, side='1P', stamp=11.)]
    assert counts(classify([row], terminals, 0., 20.))['later_death_not_observed'] == 1
    terminals.append(dict(game=1, side='2P', stamp=12.))
    assert counts(classify([row], terminals, 0., 20.))['later_death_observed'] == 1
