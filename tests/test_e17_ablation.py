"""各アブレーションのフラグ独立性と既定OFFを検証する。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src.board import Board
from src.exchange_event_overlay import ConfirmedSide, ExchangeEventOverlay
from scripts.run_e17_ablation_20260928 import VARIANTS


@pytest.mark.parametrize("variant,expected", [
    ("sync", (True, False, False, False)), ("death", (False, True, False, False)),
    ("layers", (False, False, True, False)), ("all", (True, True, True, False)),
    ("fixed", (False, False, True, True)),
])
def test_independent_flags(variant: str, expected: tuple) -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, **VARIANTS[variant])
    layer = overlay._e16
    assert (layer.sync_enabled, layer.death_enabled, layer.layer_enabled, layer.completion_fix) == expected
    assert overlay.tracker.live_count


def test_default_is_off() -> None:
    overlay = ExchangeEventOverlay(NS(), None, None)
    assert overlay._e16 is None
    assert not overlay.tracker.live_count


@pytest.mark.parametrize("variant", ["death", "layers", "fixed"])
def test_no_sync_gate_without_sync_flag(variant: str) -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, **VARIANTS[variant])
    overlay._history = [[ConfirmedSide(1., Board(), np.ones(4, int))] for _ in range(2)]
    assert overlay._count_inputs() == [h[-1] for h in overlay._history]


@pytest.mark.parametrize("variant", ["sync", "layers", "fixed"])
def test_no_death_rejection_without_death_flag(variant: str) -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, **VARIANTS[variant])
    assert not overlay._e16.before(overlay, NS(confirmed_dead_sides=("2P",)), 1.)


def test_death_only_does_not_blend() -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, death_guard=True)
    overlay.tracker.probability = .7
    overlay._e16.apply(overlay, None, None, 1.)
    assert overlay.tracker.probability == .7


def test_layers_observe_death_without_rejecting_fire() -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, evaluation_layers=True)
    assert not overlay._e16.before(overlay, NS(confirmed_dead_sides=("2P",)), 1.)
    assert overlay._e16.dead_sides == {"2P"}
