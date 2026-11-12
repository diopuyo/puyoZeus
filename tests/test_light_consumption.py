"""NEXT記帳と隠し段の印の局所試験。"""
from __future__ import annotations

import pytest

from src.light_consumption import LightConsumptionTracker, enabled, supported_ojama


def packet(t: float, pair: tuple | None, state: str = 'STABLE', board: list | None = None,
           game: int = 0) -> dict:
    grid = board or [[0] * 6 for _ in range(13)]
    side = dict(state=dict(state=state), confirmed_board=grid, raw_cnn_board=grid,
                tape_next_pair=pair, tsumo_count=10, formula=dict(valid=False))
    return dict(t=t, game=game, raw=[side, dict(side)])


def confirm(tracker: LightConsumptionTracker, t: float, pair: tuple,
            state: str = 'STABLE', board: list | None = None) -> dict:
    for offset in (0., .034, .068):
        result = tracker.observe(packet(t+offset, pair, state, board))
    return result['raw'][0]


def test_state_does_not_drop_consumption() -> None:
    tracker = LightConsumptionTracker()
    confirm(tracker, 0, (1, 2))
    result = confirm(tracker, 1, (2, 3), 'OJAMA_FALL')
    assert result['tsumo_count'] == 10 and result['light_next_consumed']
    assert confirm(tracker, 2, (2, 3))['light_tape_advances'] == 1


def test_jitter_and_missing_are_not_consumption() -> None:
    tracker = LightConsumptionTracker()
    confirm(tracker, 0, (1, 2))
    tracker.observe(packet(.2, (2, 3)))
    tracker.observe(packet(.24, None))
    assert confirm(tracker, .3, (1, 2))['tsumo_count'] == 10


def test_hidden_mark_waits_for_unchanged_settled_board() -> None:
    tracker = LightConsumptionTracker()
    full = [[0]*6] + [[1, 0, 0, 0, 0, 0] for _ in range(12)]
    confirm(tracker, 0, (1, 2), board=full)
    assert not confirm(tracker, 1, (2, 3), board=full)['light_hidden_inflow_unknown']
    result = tracker.observe(packet(1.6, (2, 3), board=full))['raw'][0]
    assert result['light_hidden_inflow_unknown']
    assert not tracker.observe(packet(2, (2, 3)))['raw'][0]['light_hidden_inflow_unknown']


def test_visible_landing_cancels_hidden_mark() -> None:
    tracker = LightConsumptionTracker()
    full = [[0]*6] + [[1, 0, 0, 0, 0, 0] for _ in range(12)]
    confirm(tracker, 0, (1, 2), board=full)
    confirm(tracker, 1, (2, 3), board=full)
    full[-1][1] = 2
    full[-2][1] = 3
    assert not tracker.observe(packet(1.6, (2, 3), board=full))['raw'][0]['light_hidden_inflow_unknown']


def test_real_ojama_is_supported_even_during_opponent_chain() -> None:
    before = (0,)*72
    after = list(before)
    after[-1] = 9
    assert supported_ojama(before, tuple(after)) == 1
    after[-1], after[0] = 0, 9
    assert supported_ojama(before, tuple(after)) == 0


def test_game_reset_and_duplicate_packet() -> None:
    tracker = LightConsumptionTracker()
    confirm(tracker, 0, (1, 2))
    confirm(tracker, 1, (2, 3))
    assert tracker.observe(packet(1.068, (2, 3)))['raw'][0]['light_tape_advances'] == 1
    assert tracker.observe(packet(2, (2, 3), game=1))['raw'][0]['tsumo_count'] == 10


def test_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIGHT_ROBUST_CELLS', raising=False)
    assert not enabled()
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    assert enabled()


@pytest.mark.parametrize('flag,marked,predicted', [('0', True, True), ('1', False, True), ('1', True, False)])
@pytest.mark.parametrize('module', ['src.fire_board', 'src.phase_j._live_ex_frozen.src.fire_board'])
def test_hidden_mark_suppresses_only_enabled_fire_prediction(
    monkeypatch: pytest.MonkeyPatch, flag: str, marked: bool, predicted: bool, module: str,
) -> None:
    from importlib import import_module
    PairForecast = import_module(module).PairForecast
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', flag)
    forecast = PairForecast()
    grid = [[0]*6 for _ in range(13)]
    grid[-1][:4] = [1]*4
    raw = dict(raw_cnn_board=grid, formula=dict(valid=False),
               light_hidden_inflow_unknown=marked)
    forecast.observe(.9, 0, [raw, dict(raw)])
    raw['formula'] = dict(valid=True, product=40)
    forecast.observe(1., 0, [raw, dict(raw)])
    result = forecast.current(0, 1.)
    assert result is not None and result.has_prediction == predicted
    assert result.observed == (40,)


def test_pipeline_publication_and_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import Counter
    from types import SimpleNamespace
    from src.recognition_pipeline import RecognitionPipeline
    from src.light_consumption import annotate_sides
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    pipeline = object.__new__(RecognitionPipeline)
    pipeline._tsumo_count_1p = Counter({1: 20})
    pipeline._tsumo_count_2p = Counter({1: 20})
    owner = SimpleNamespace()
    original = [dict(state='OJAMA_FALL', tsumo_count=10, next_pair=(5, 5))]*2
    for t in (0., .034, .068):
        annotate_sides(owner, original, ((1, 2), (1, 2)), t, 0)
    for t in (1., 1.034, 1.068):
        result = annotate_sides(owner, original, ((2, 3), (1, 2)), t, 0)
    assert pipeline.tsumo_count('1P') == 10
    assert result[0]['light_next_consumed']
    assert [{k: v for k, v in s.items() if not k.startswith('light_')} for s in result] == original
    assert pipeline.tsumo_count('2P') == 10
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '0')
    assert annotate_sides(owner, original, ((2, 3), (1, 2)), 2, 0) is original
    assert pipeline.tsumo_count('1P') == 10


def test_partial_hidden_and_first_step_release() -> None:
    tracker = LightConsumptionTracker()
    full = [[0]*6] + [[1, 0, 0, 0, 0, 0] for _ in range(12)]
    confirm(tracker, 0, (1, 2), board=full)
    confirm(tracker, 1, (2, 3), board=full)
    full[-1][1] = 2
    assert tracker.observe(packet(1.6, (2, 3), board=full))['raw'][0]['light_hidden_inflow_unknown']
    fire = packet(2, (2, 3), 'CHAIN', full)
    fire['raw'][0]['formula'] = dict(valid=True, product=40)
    assert tracker.observe(fire)['raw'][0]['light_hidden_inflow_unknown']
    after = tracker.observe(packet(2.9, (2, 3), 'CHAIN', full))['raw'][0]
    assert not after['light_hidden_inflow_unknown']
    assert after['light_hidden_clear_reason'] == 'first_step_finished'


@pytest.mark.parametrize('value', [0, 9, 10])
def test_chain_garbage_unknown_changes_are_not_hidden_placement(value: int) -> None:
    tracker = LightConsumptionTracker()
    full = [[0]*6] + [[1, 0, 0, 0, 0, 0] for _ in range(12)]
    confirm(tracker, 0, (1, 2), board=full)
    confirm(tracker, 1, (2, 3), board=full)
    full[-1][0] = value
    assert not tracker.observe(packet(1.6, (2, 3), board=full))['raw'][0]['light_hidden_inflow_unknown']


@pytest.mark.parametrize('module', ['src.fire_board', 'src.phase_j._live_ex_frozen.src.fire_board'])
def test_guard_does_not_change_history_after_chain(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import import_module
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    cls = import_module(module).PairForecast
    clean, guarded = cls(), cls()
    grid = [[0]*6 for _ in range(13)]
    grid[-1][:4] = [1]*4
    for t, valid, consumed in ((.9, False, False), (1., True, False), (1.9, False, True)):
        raw = dict(raw_cnn_board=grid, formula=dict(valid=valid, product=40 if valid else None),
                   light_next_consumed=consumed)
        clean.observe(t, 0, [raw, dict(raw)])
        guarded.observe(t, 0, [dict(raw, light_hidden_inflow_unknown=True), dict(raw)])
    a, b = clean.current(0, 1.9), guarded.current(0, 1.9)
    assert a.chain_id == b.chain_id and a.steps == b.steps and a.source == b.source


def test_b3_guard_does_not_fall_back_to_small_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    import math
    from types import SimpleNamespace
    from src.b3_state import SideState, Episode
    from src.b3_chain_prediction import EpisodePrediction
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    state = object.__new__(SideState)
    origin = EpisodePrediction(score=40., length=1.)
    state.episode = Episode(1., 1., prediction=origin)
    fc = SimpleNamespace(observed=(40,), chain_id=1., has_prediction=False, final_board=None)
    state.forecast = SimpleNamespace(_light_hidden_fire=True, current=lambda: fc)
    state._apply_forecast(1.1)
    assert math.isnan(state.episode.prediction.score)
    assert state.episode.prediction.reason == 'hidden_inflow_unknown'
    state.forecast._light_hidden_fire = False
    state._apply_forecast(2.)
    assert state.episode.prediction is origin
