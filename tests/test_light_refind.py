"""未知升の選択・下限表示・式矛盾時の復帰を小さい盤で検証する。"""
from __future__ import annotations
from importlib import import_module
from types import SimpleNamespace
import numpy as np
import pytest
from src.fire_board import NativeTrace
from src.light_consumption import annotate_sides, guarded_forecast
from src.light_refind import PhysicalRefind, after_first, physical_forecast, update_physical_refind

MODULES = ('src.fire_board', 'src.phase_j._live_ex_frozen.src.fire_board')


def test_b3_annotation_supplies_separate_next_pair_to_physical_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    owner, predictor = SimpleNamespace(), PhysicalRefind(NativeTrace())
    source = [dict(state='STABLE', confirmed_board=full_column(), tsumo_count=7,
                   next_pair=(5, 5)) for _ in range(2)]
    for t in (.1, .2, .3):
        sides = annotate_sides(owner, source, ((1, 4), (2, 5)), t, 0)
        predictor.remember(t, sides[0])
    assert list(predictor.pairs) == [(.3, (1, 4))]
    assert sides[0]['next_pair'] == (5, 5)
    assert sides[0]['tsumo_count'] == 7
    assert 'light_next_pair' not in source[0]


def full_column() -> np.ndarray:
    board = np.zeros((13, 6), np.int8)
    board[1:, 0] = (2, 4, 5, 2, 4, 5, 2, 4, 1, 1, 1, 1)
    return board


def prepared() -> tuple[PhysicalRefind, np.ndarray, SimpleNamespace]:
    predictor = PhysicalRefind(NativeTrace())
    before = full_column()
    predictor.remember(.1, dict(state={'state': 'STABLE'}, confirmed_board=before))
    forecast = SimpleNamespace(chain_id=1., observed=(40,))
    predictor.observe(1., dict(formula={'valid': True, 'product': 40}), forecast, True)
    after, slots = after_first(before, 40, {0})
    assert slots == [(4, 0)]
    return predictor, after, forecast


def observe_probs(predictor: PhysicalRefind, raw: np.ndarray, forecast: object) -> None:
    codes = np.where(raw[1:] == 9, 6, raw[1:])
    probs = np.eye(7)[codes]*.99+.01/7
    for t in np.linspace(1.9, 2.2, 10):
        predictor.observe(float(t), dict(light_cell_probs=probs, formula={'valid': False}), forecast, True)


def test_first_pop_does_not_invent_hidden_puyo_above_newly_full_column() -> None:
    after, slots = after_first(full_column(), 40, set())
    assert slots == []
    assert np.count_nonzero(after) == 8


def test_unknown_keeps_candidate_set_without_selecting_board() -> None:
    predictor, after, forecast = prepared()
    assert len(predictor.candidates) == 5
    raw = after.copy()
    raw[4, 0], raw[12, 3] = 5, 9
    observe_probs(predictor, raw, forecast)
    assert predictor.best is None
    assert predictor.evidence.steps == (40,)
    kept = [predictor.candidates[i] for i in predictor.evidence.retained]
    assert all(c.board[4, 0] == 5 and c.board[12, 3] == 0 for c in kept)
    assert all(not c.board[0].any() for c in kept)


def test_equal_probability_evidence_keeps_both_candidates() -> None:
    predictor, after, forecast = prepared()
    for index, color in enumerate((1, 2)*5):
        raw = after.copy()
        raw[4, 0] = color
        probs = np.eye(7)[raw[1:]]*.99+.01/7
        predictor.observe(1.9+index*.034, dict(light_cell_probs=probs, formula={'valid': False}), forecast, True)
    assert predictor.best is None
    assert {predictor.candidates[i].board[4,0] for i in predictor.evidence.retained} == {1,2}
    assert predictor.evidence.final is None


@pytest.mark.parametrize('module', MODULES)
def test_second_formula_conflict_returns_original_forecast(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    predictor, after, forecast = prepared()
    observe_probs(predictor, after, forecast)
    assert predictor.evidence.steps == (40,)
    original = import_module(module).Forecast(1., (40, 320), (40, 320, 900), 'refind', after, 2.4)
    owner = SimpleNamespace(_light_physics=predictor)
    assert physical_forecast(owner, original) is original
    predictor.observe(2.4, dict(formula={'valid': True, 'product': 320}), original, True)
    assert predictor.rejected and predictor.best is None
    assert physical_forecast(owner, original) is original
    assert original.steps == (40, 320, 900)


@pytest.mark.parametrize('module', MODULES)
def test_lower_bound_copy_is_explicit_and_does_not_change_internal_forecast(
    module: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    predictor, after, forecast = prepared()
    observe_probs(predictor, after, forecast)
    original = import_module(module).Forecast(1., (40,), (40,), 'none', after, 1.)
    value = physical_forecast(SimpleNamespace(_light_physics=predictor), original)
    assert value is not original and value.lower_bound and value.source == 'light_physics_lower'
    assert not original.lower_bound and original.source == 'none'
    assert guarded_forecast(value, True) is value


@pytest.mark.parametrize('module', MODULES)
def test_guard_preserves_visible_continuation(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    original = import_module(module).Forecast(1., (40,), (40, 320), 'fire_board', full_column(), 1.)
    assert guarded_forecast(original, True) is original


def test_off_does_not_allocate_physics_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PUYO_LIGHT_ROBUST_CELLS', raising=False)
    owner = SimpleNamespace()
    update_physical_refind(owner, 1., full_column(), {})
    assert not hasattr(owner, '_light_physics')


@pytest.mark.parametrize('module', MODULES)
def test_reset_discards_physics_state(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIGHT_ROBUST_CELLS', '1')
    owner = import_module(module).ChainForecast()
    owner.observe(.1, {'valid': False}, full_column(), {})
    assert hasattr(owner, '_light_physics')
    owner.reset()
    assert not hasattr(owner, '_light_physics')
