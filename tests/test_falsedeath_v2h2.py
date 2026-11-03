"""D-v2h2の事前登録にある開始・独立照合・鮮度・配線を検証する。"""
from __future__ import annotations

from dataclasses import make_dataclass
from types import SimpleNamespace as NS
from typing import Any

import pytest

from src.board_state_machine import BoardState
from src.exchange_history_integrity import BOUNDARY_ZERO_MAX_SEC, MatchHistoryIntegrity
from tests.test_falsedeath_v2d import sides


def start_guard() -> MatchHistoryIntegrity:
    """前試合の非ゼロ得点と連続した正式境界を用意する。"""
    guard = MatchHistoryIntegrity(boundary_score_start=True, strict_integrity=True)
    pair = sides(20, 20)
    for side in pair:
        side.score = 1000
    guard.observe(pair, 0, 0.)
    return guard


@pytest.mark.parametrize('count', [0, 1, 2])
def test_start_requires_small_stable_boards(count: int) -> None:
    guard = start_guard()
    guard.observe(sides(count, count), 1, 1.)
    assert guard.evidence()['history_complete']


@pytest.mark.parametrize('problem', ['menu', 'chain', 'missing', 'occupied'])
def test_unconfirmed_boundary_never_accepts(problem: str) -> None:
    guard = start_guard()
    pair = sides(2, 2)
    if problem == 'missing':
        pair[1].confirmed_board = None
    elif problem == 'occupied':
        pair = sides(2, 3)
    else:
        pair[1].state = BoardState.MENU if problem == 'menu' else BoardState.CHAIN
    guard.observe(pair, 1, 1.)
    assert not guard.evidence()['history_complete']


@pytest.mark.parametrize('delay,accepted', [(BOUNDARY_ZERO_MAX_SEC, True),
                                         (BOUNDARY_ZERO_MAX_SEC+0.001, False)])
def test_first_zero_deadline(delay: float, accepted: bool) -> None:
    guard = start_guard()
    pair = sides(2, 2)
    for side in pair:
        side.score = None
    guard.observe(pair, 1, 1.)
    guard.observe(sides(2, 2), 1, 1.+delay)
    assert guard.evidence()['history_complete'] is accepted


def test_frozen_live_and_replay_history_agree() -> None:
    guards = [start_guard(), start_guard()]
    pair = sides(2, 2)
    live_type = make_dataclass('LiveSide', [(k, Any) for k in vars(pair[0])], frozen=True)
    guards[0].observe(pair, 1, 1.)
    guards[1].observe(tuple(live_type(**vars(s)) for s in pair), 1, 1.)
    assert guards[0].summary() == guards[1].summary()


def test_history_new_parameters_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        MatchHistoryIntegrity(True)


def strict_case() -> tuple[Any, ...]:
    """既存v2hの正対照の予測を、独立した発火時記録へ移す。"""
    from copy import deepcopy
    from tests.test_falsedeath_v2h import recent_case
    case = recent_case()
    engine, projection, _, chain, _ = case
    engine.strict_integrity = True
    projection.post_counter_bound = engine
    engine.base_predictions[chain.chain_id] = deepcopy(projection.safety.recovery.entries[1]['options'][0])
    return case


@pytest.mark.parametrize('kind', ['hidden', 'recovery', 'midchain'])
@pytest.mark.parametrize('base_matches', [False, True])
def test_only_base_prediction_is_compared(kind: str, base_matches: bool) -> None:
    from copy import deepcopy
    from src.exchange_post_counter_recent import check_recent
    engine, projection, _, chain, _ = strict_case()
    base = engine.base_predictions[1]
    adapted = deepcopy(base)
    (adapted if base_matches else base)['prefix'][8] += 1
    hidden = None
    if kind == 'hidden':
        hidden = dict(options=[adapted])
    elif kind == 'recovery':
        projection.safety.recovery.entries[1]['options'] = [adapted]
    else:
        projection.safety.recovery.entries.clear()
        projection.midchain = NS(entries={1: dict(chain=chain, active=adapted)})
    check = check_recent(projection, chain, hidden, engine.step_observations[(0, 1)],
                         strict_integrity=True, stamp=1.)
    assert check['accepted'] is base_matches
    assert len(check['candidates']) == 1


def test_adapted_candidates_cannot_replace_missing_base() -> None:
    from tests.test_falsedeath_v2f import prove
    case = strict_case()
    case[0].base_predictions.clear()
    assert not prove(case)['dead']


@pytest.mark.parametrize('lag,accepted', [(0, True), (1, True), (2, False), (3, False)])
def test_stale_two_stages_is_unobserved(lag: int, accepted: bool) -> None:
    from src.exchange_prefire_stage_timeout import STAGE_INTERVAL_SEC
    case = strict_case()
    engine, projection, overlay, chain, context = case
    stamp = (8+lag)*STAGE_INTERVAL_SEC
    result = engine.prove(projection, overlay, 0, (), 171, 1, context, stamp)
    check = result['completion_prefix_check']
    assert check['accepted'] is accepted
    assert check['stage_lag'] == lag
    if not accepted:
        assert check['reason'] == 'recent_unobserved' and not result['dead']


@pytest.mark.parametrize('conflict', ['stage_regression', 'formula_revision', 'invalid_stage'])
def test_conflicts_survive_later_matching_window(conflict: str) -> None:
    from src.exchange_post_counter_steps import record_formula
    from tests.test_falsedeath_v2f import prove
    case = strict_case()
    entry = case[0].step_observations[(0, 1)]
    entry.update(stages={}, conflicts=[], last_count=None)
    for count, score in [(2, 420), (1, 100)] if conflict == 'stage_regression' else (
            [(1, 100), (1, 101)] if conflict == 'formula_revision' else [(0, 0)]):
        record_formula(entry, count, score)
    for count, score in [(7, 15500), (8, 21900), (9, 29580)]:
        record_formula(entry, count, score)
    assert conflict in entry['conflicts']
    assert not prove(case)['dead']
    assert case[0].audit[-1]['completion_prefix_check']['reason'] == 'recent_mismatch'


def test_landing_passes_real_formula_result_before_projection(monkeypatch: Any) -> None:
    from src.exchange_event_landing import ExchangeLandingProjection
    from src.chain_detector import CHAIN_MECHANISM_FORMULA_READ
    engine, _, overlay, chain, _ = strict_case()
    engine.step_observations.clear()
    projection = ExchangeLandingProjection()
    projection.post_counter_bound = engine
    overlay._landing_projection = projection
    side = NS(chain_event=NS(mechanism=CHAIN_MECHANISM_FORMULA_READ,
        trigger_sec=0., chain_count=9, total_score=29580))
    result = NS(p1=side, p2=NS(chain_event=None))
    monkeypatch.setattr(projection, '_observe_frame', lambda *args: None)
    def checked_before_return(*args: Any) -> bool:
        assert engine.step_observations[(0, 1)]['stages'][9]['formula'] == 29580
        return True
    monkeypatch.setattr(projection, '_refresh_death', checked_before_return)
    projection.update(overlay, result, NS(), 1.)


def test_freshness_invalidates_outer_cache_without_new_formula() -> None:
    from src.exchange_prefire_stage_timeout import STAGE_INTERVAL_SEC
    engine, projection, overlay, _, _ = strict_case()
    engine.refresh_current_observation(overlay, stamp=9*STAGE_INTERVAL_SEC)
    revision = engine.revision
    engine.refresh_current_observation(overlay, stamp=10*STAGE_INTERVAL_SEC)
    assert engine.revision > revision


def test_flags_default_off_and_keyword_only() -> None:
    import inspect
    from src.exchange_post_counter_bound import PostCounterDeathBound
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    from scripts.falsedeath_replay_20261001 import enabled_flags
    for constructor in (MatchHistoryIntegrity, PostCounterDeathBound):
        param = inspect.signature(constructor).parameters['strict_integrity']
        assert param.default is False and param.kind == inspect.Parameter.KEYWORD_ONLY
    for name, param in inspect.signature(PostCounterDeathBound).parameters.items():
        if name != 'early_exit':
            assert param.kind == inspect.Parameter.KEYWORD_ONLY
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['death_integrity_v2h2'].default is False
    assert enabled_flags('v2h2') == enabled_flags('v2h') + ['--death-integrity-v2h2']


@pytest.mark.parametrize('missing', ['death_history_integrity', 'death_history_boundary_start',
                                   'post_counter_completion_recent_integrity'])
def test_strict_cli_dependencies(missing: str) -> None:
    import argparse
    from src.exchange_event_cli import validate_v2_options
    options = dict(death_integrity_v2h2=True, death_history_integrity=True,
        death_history_boundary_start=True, post_counter_completion_recent_integrity=True,
        post_counter_death_bound=True, post_counter_current_puyo_integrity_v2c=True)
    options[missing] = False
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(**options))


def test_boundary_start_alone_is_rejected() -> None:
    import argparse
    from src.exchange_event_cli import validate_v2_options
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(death_history_boundary_start=True))


def test_result_wiring_mutation_is_killed(monkeypatch: Any) -> None:
    """resultを渡さない変異が、実updateを通る配線テストで落ちることも確認する。"""
    import inspect
    import textwrap
    from src import exchange_event_landing as module
    original = inspect.getsource(module.ExchangeLandingProjection.update)
    broken = original.replace('refresh_current_observation(overlay, result, t_sec)',
                              'refresh_current_observation(overlay, None, t_sec)')
    assert broken != original
    namespace = dict(vars(module))
    exec(compile(textwrap.dedent(broken), '<result配線削除>', 'exec'), namespace)
    monkeypatch.setattr(module.ExchangeLandingProjection, 'update', namespace['update'])
    with pytest.raises((AssertionError, KeyError)):
        test_landing_passes_real_formula_result_before_projection(monkeypatch)


@pytest.mark.parametrize('state', [BoardState.STABLE, BoardState.MENU, BoardState.CHAIN])
def test_shared_overlay_h1_accepts_same_live_and_replay_input(state: BoardState) -> None:
    """共有overlayの実配線を通し、ライブの凍結入力でも開始受理が一致する。"""
    from src.exchange_event_overlay import ExchangeEventOverlay
    overlays = [ExchangeEventOverlay.__new__(ExchangeEventOverlay) for _ in range(2)]
    for overlay in overlays:
        overlay.tracker, overlay._prefire = NS(), None
        overlay._initialize_state(True, True, True)
    previous = sides(20, 20)
    for side in previous:
        side.score = 1000
    pair = sides(2, 2)
    pair[1].state = state
    live_type = make_dataclass('LiveSide', [(k, Any) for k in vars(pair[0])], frozen=True)
    for game, stamp, row in [(0, 0., previous), (1, 1., pair)]:
        overlays[0]._observe_prefire(row, stamp, game)
        overlays[1]._observe_prefire(tuple(live_type(**vars(s)) for s in row), stamp, game)
    assert overlays[0]._history_integrity.summary() == overlays[1]._history_integrity.summary()
    assert overlays[1]._history_integrity.evidence()['history_complete'] is (state == BoardState.STABLE)
