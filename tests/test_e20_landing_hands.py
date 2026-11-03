"""E20のゲーム仕様・因果性・観測不足を小さな物理例で検証する。"""
from types import SimpleNamespace as NS
import pytest
from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_event_hands import LandingHandsObservation, spec_hands


@pytest.mark.parametrize("remaining,interval,expected", [(0., .7, 1), (.69, .7, 1),
    (.7, .7, 2), (2.1, 1., 3), (2.1, 2., 2), (-1., .5, 1)])
def test_spec_formula(remaining: float, interval: float, expected: int) -> None:
    assert spec_hands(remaining, interval) == expected


@pytest.mark.parametrize("interval", [0., -1., float("nan"), float("inf")])
def test_no_invented_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        spec_hands(1., interval)


def side(count: int = 0, motion: bool = False, event: object = None, shifted: bool = False) -> NS:
    board = Board()
    board._grid[-1, :count] = 1
    return NS(state=BoardState.STABLE, confirmed_board=board, next_slide_motion=motion, chain_event=event,
              next_pair=(2, 2) if shifted else (1, 1), dnext_pair=(3, 3) if shifted else (2, 2))


def test_partial_board_and_next_edges_do_not_double_count() -> None:
    state = LandingHandsObservation()
    for stamp, count, motion in [(0., 0, False), (.1, 0, False), (1., 1, False), (1.1, 2, False),
                                 (1.2, 2, False), (1.3, 2, True), (1.4, 2, False), (2., 4, False)]:
        state.observe((side(count, motion, shifted=stamp >= 1.3), side()), stamp)
    assert state.placements[0] == [1.1, 2.]
    assert list(state.intervals[0]) == pytest.approx([.9])


def test_next_start_confirms_without_waiting_for_stable_or_end_signal() -> None:
    state = LandingHandsObservation()
    chain = NS(trigger_sec=2., predicted_chain_count=4, end_signal_sec=None)
    tracker = NS(latest_chain=lambda label: chain if label == "1P" else None)
    state.observe((side(), side()), 1.8)
    state.observe((side(), side()), 1.9)
    state.observe_chains(tracker, [1, 0], 2.)
    moving = side(motion=True, shifted=True)
    moving.state = BoardState.GRAVITY_SETTLE
    state.observe((moving, side()), 2.6)
    state.observe((moving, side()), 2.7)
    state.observe_chains(tracker, [1, 0], 2.7)
    assert state.estimate(chain, 0, 2.6, 1)["hands"] == 1
    assert state.estimate(chain, 0, 2.6, 1)["reason"] == "next_motion_confirmed"
    assert list(state.animations[1]) == pytest.approx([.6])


def test_recent_median_and_game_reset() -> None:
    state = LandingHandsObservation()
    state.animations[2].extend([1., 1.2, 1.4])
    state.intervals[1].extend([.4, .5, 2.])
    chain = NS(trigger_sec=10., predicted_chain_count=2)
    assert state.estimate(chain, 0, 10.1, 1)["hands"] == 3
    state.reset()
    assert state.estimate(chain, 0, 10.1, 1)["reason"] == "missing_placement_interval"
    assert list(state.animations[2]) == [1., 1.2, 1.4]


@pytest.mark.parametrize("attacker", [0, 1])
@pytest.mark.parametrize("confirmed", [True, False])
def test_side_selection_and_immediate_confirmation(attacker: int, confirmed: bool) -> None:
    state = LandingHandsObservation()
    state.animations[2].append(2.)
    state.intervals[attacker].append(100.)
    state.intervals[1-attacker].append(.5)
    chain = NS(trigger_sec=10., predicted_chain_count=2)
    if confirmed:
        state.completed[attacker][10.] = 10.1
    assert state.estimate(chain, attacker, 10.2, 2)["hands"] == (1 if confirmed else 4)


@pytest.mark.parametrize("have_animation,have_interval,reason", [
    (False, False, "missing_animation"), (False, True, "missing_animation"),
    (True, False, "missing_placement_interval")])
def test_missing_observations_never_use_old_time_table(
        have_animation: bool, have_interval: bool, reason: str) -> None:
    state = LandingHandsObservation()
    if have_animation:
        state.animations[2].append(3.)
    if have_interval:
        state.intervals[1].append(.5)
    result = state.estimate(NS(trigger_sec=1., predicted_chain_count=2), 0, 1.1, 2)
    assert result["hands"] == 1 and result["reason"] == reason


def test_recent_observations_only_and_median_not_mean() -> None:
    state = LandingHandsObservation()
    state.animations[2].extend([10., 1., 1., 1., 2., 2.])
    state.intervals[1].extend([100., .5, .5, .5, 10., 10.])
    result = state.estimate(NS(trigger_sec=1., predicted_chain_count=2), 0, 1., 2)
    assert result["hands"] == 3 and len(result["placement_intervals"]) == 5


def test_animation_not_available_until_next_moves() -> None:
    state = LandingHandsObservation()
    chain = NS(trigger_sec=1., predicted_chain_count=2)
    tracker = NS(latest_chain=lambda label: chain if label == "1P" else None)
    state.observe((side(), side()), .8)
    state.observe((side(), side()), .9)
    state.observe_chains(tracker, [2, 0], 1.)
    assert not state.animations[2]
    state.observe((side(motion=True, shifted=True), side()), 2.)
    state.observe((side(shifted=True), side()), 2.1)
    state.observe_chains(tracker, [2, 0], 2.1)
    assert list(state.animations[2]) == [1.]
    state.observe((side(motion=True, shifted=True), side()), 3.)
    assert list(state.animations[2]) == [1.]
    assert state.estimate(chain, 0, 3., 2)["hands"] == 1


def test_attack_flash_without_queue_shift_never_confirms_chain() -> None:
    state = LandingHandsObservation()
    tracker = NS(latest_chain=lambda label: NS(trigger_sec=1.) if label == "1P" else None)
    for stamp, motion in [(.8, False), (.9, False), (1.5, True), (1.6, False), (1.7, False)]:
        state.observe((side(motion=motion), side()), stamp)
        if stamp > 1.:
            state.observe_chains(tracker, [2, 0], stamp)
    assert not state.completed[0] and not state.animations[2]


@pytest.mark.parametrize("end,expected", [(2., True), (20., False)])
def test_validated_identical_color_next_fallback_is_causal(end: float, expected: bool) -> None:
    state = LandingHandsObservation()
    chain = NS(trigger_sec=1., end_confirmed=True, end_reason="tsumo", end_signal_sec=end)
    state.observe_chains(NS(latest_chain=lambda label: chain if label == "1P" else None), [2, 0], 3.)
    assert bool(state.completed[0]) is expected


def test_missing_lower_bound_cannot_prove_death() -> None:
    from src.exchange_event_landing import ExchangeLandingProjection
    projection = ExchangeLandingProjection(hands_spec=True)
    tracker = NS(latest_chain=lambda _: NS(trigger_sec=1., predicted_chain_count=2))
    assert not projection._known_budget(tracker, 0, 1.1)
    projection.hands_observation.completed[0][1.] = 1.1
    assert projection._known_budget(tracker, 0, 1.1)


@pytest.mark.parametrize("function", ["overlay", "replay", "generate"])
def test_public_flag_defaults_off(function: str) -> None:
    import inspect
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    value = {"overlay": ExchangeEventOverlay, "replay": replay, "generate": generate}[function]
    assert inspect.signature(value).parameters["landing_hands_spec"].default is False


@pytest.mark.parametrize("hands,level", [(1, -1), (2, 0), (3, 1), (6, 4)])
def test_nf_search_budget_includes_known_pairs_once(
        monkeypatch: pytest.MonkeyPatch, hands: int, level: int) -> None:
    from src import exchange_event_landing as landing
    calls = []
    def search(*args: object, **kwargs: object) -> NS:
        calls.append(kwargs["k_levels"])
        return NS(values={level: NS(raw=0)})
    monkeypatch.setattr(landing, "near_future_fire_power", search)
    landing.future_send.cache_clear()
    grid = Board()._grid
    landing.future_send(grid.tobytes(), grid.shape, grid.dtype.str, (1, 2, 3, 4), hands, 0.)
    assert calls == [(level,)]
    landing.future_send.cache_clear()
