"""観測済み死亡の確定表示、OFF互換、試合境界とEMA非混入を検証する。"""
from types import SimpleNamespace as NS
import pytest
from src.exchange_event_overlay import ExchangeEventOverlay
from src.exchange_event_terminal import confirmed_winner_probability
from scripts.visualize_advantage_overlay import _exchange_display, _ExchangeDisplayEMA


@pytest.mark.parametrize("dead,expected", [(set(), None), ({"1P"}, 0.), ({"2P"}, 1.), ({"1P", "2P"}, None)])
def test_existing_terminal_direction(dead: set[str], expected: float | None) -> None:
    assert confirmed_winner_probability(dead) == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_death_guard_freeze_becomes_exact_only_when_enabled(enabled: bool) -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, death_guard=True, confirmed_death_hold=enabled)
    overlay._game = 2
    overlay.tracker.probability, overlay.tracker.source = .764, "S3_landing"
    result = NS(confirmed_dead_sides=("2P",), p1=NS(chain_event=None), p2=NS(chain_event=None))
    overlay.update(result, None, None, 2698.98, 2)
    assert overlay.tracker.probability == (1. if enabled else .764)
    result.confirmed_dead_sides = ()
    overlay.update(result, None, None, 2700., 2)
    assert overlay.tracker.probability == (1. if enabled else .764)


def test_boundary_clears_confirmed_winner() -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, confirmed_death_hold=True)
    overlay._e16.dead_sides = {"2P"}
    overlay._hold_confirmed_death(1.)
    overlay._reset(3, 2.)
    assert not overlay._e16.dead_sides and overlay.tracker.probability is None


def test_exact_display_does_not_smooth_or_pollute_next_game() -> None:
    overlay = NS(tracker=NS(probability=1., source="confirmed_death"))
    smoothing = _ExchangeDisplayEMA(adv=20., probability=.6, last_sec=1.)
    assert _exchange_display(overlay, 0., .5, smoothing, 2.) == (100., 1.)
    assert (smoothing.adv, smoothing.probability, smoothing.last_sec) == (20., .6, 1.)


def test_flag_does_not_enable_sync_or_prediction_blend() -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, confirmed_death_hold=True)
    assert overlay._e16.death_enabled
    assert not overlay._e16.layer_enabled and not overlay._e16.sync_enabled
    assert not ExchangeEventOverlay(NS(), None, None)._confirmed_death_hold
