"""両者共通の設置起点と欠測・試合境界を検証する。"""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.margin_clock import MarginClock, PlacementObservation, first_placement_origin


def board(values: tuple[int, ...] = ()) -> Board:
    """最下段に指定色を並べた独立盤面を作る。"""
    result = Board()
    result._grid[-1, :len(values)] = values
    return result


def side(values: tuple[int, ...] = (), state: BoardState = BoardState.STABLE) -> SimpleNamespace:
    """観測に必要な属性だけ用意する。"""
    return SimpleNamespace(state=state, confirmed_board=board(values))


@pytest.mark.parametrize("times", [(5., 7.), (7., 5.), (5., 5.), (0., 1.)])
def test_earliest_side_and_input_order(times: tuple[float, float]) -> None:
    observations = [PlacementObservation(t, BoardState.STABLE, board(), board((1, 2))) for t in times]
    assert first_placement_origin(observations) == min(times)
    assert first_placement_origin(reversed(observations)) == min(times)


@pytest.mark.parametrize("values", [(), (1,), (1, 2, 3), (9, 9), (1, 10), (0, 2)])
def test_not_a_confirmed_pair(values: tuple[int, ...]) -> None:
    assert first_placement_origin([PlacementObservation(2., BoardState.STABLE, board(), board(values))]) is None


@pytest.mark.parametrize("state", [s for s in BoardState if s != BoardState.STABLE])
def test_nonstable_rejected(state: BoardState) -> None:
    assert first_placement_origin([PlacementObservation(2., state, board(), board((1, 2)))]) is None


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), -float("inf")])
def test_invalid_timestamp(stamp: float) -> None:
    assert first_placement_origin([PlacementObservation(stamp, BoardState.STABLE, board(), board((1, 2)))]) is None


def test_missing_and_existing_board_rejected() -> None:
    for before, after in ((None, board((1, 2))), (board(), None), (board((1, 2)), board((3, 4)))):
        assert first_placement_origin([PlacementObservation(2., BoardState.STABLE, before, after)]) is None
    assert first_placement_origin([]) is None


def test_inputs_unchanged_and_repeatable() -> None:
    before, after = board(), board((1, 5))
    frozen = after._grid.copy()
    observations = [PlacementObservation(2., BoardState.STABLE, before, after)]
    assert first_placement_origin(observations) == first_placement_origin(observations) == 2.
    np.testing.assert_array_equal(after._grid, frozen)
    assert not np.any(before._grid)


@pytest.mark.parametrize("early_side", [0, 1])
def test_shared_latched_origin_and_boundary(early_side: int) -> None:
    clock = MarginClock()
    clock.observe((side(), side()), 1., 1)
    sides = [side(), side()]
    sides[early_side] = side((1, 2))
    clock.observe(tuple(sides), 3., 1)
    clock.observe((side((1, 2)), side((3, 4))), 7., 1)
    assert clock.origin == 3.
    assert clock.elapsed(103., 1.) == 100.
    clock.observe((side(), side()), 200., 2)
    assert clock.origin is None
    assert clock.elapsed(201., 200.) == 1.


def test_missing_origin_fallback_is_explicit() -> None:
    clock = MarginClock()
    clock.observe((side((1, 2, 3, 4)), side((1, 2, 3, 4))), 100., 1)
    clock.observe((side(), side()), 110., 1)
    clock.observe((side((1, 2)), side((3, 4))), 120., 1)
    assert clock.origin is None
    assert clock.elapsed(150., 100.) == 50.
    assert clock.elapsed(150., None) == 0.


def test_missing_earlier_opponent_must_not_use_later_side() -> None:
    clock = MarginClock()
    clock.observe((side((1, 2)), side()), 1., 1)
    clock.observe((side((1, 2)), side((3, 4))), 2., 1)
    assert clock.origin is None


@pytest.mark.parametrize("enabled", [False, True])
def test_accounting_rate_uses_shared_origin(enabled: bool) -> None:
    from src.ojama_accounting import OjamaAccountingTracker
    tracker = OjamaAccountingTracker(margin_origin_first_placement=enabled)
    tracker.reset(match_start_sec=10.)
    tracker.observe_margin((side(), side()), 10., 1)
    tracker.observe_margin((side(), side((1, 2))), 14., 1)
    tracker.observe_margin((side((3, 4)), side((1, 2))), 16., 1)
    assert tracker.get_effective_rate(108.) == (70 if enabled else 52)
    assert tracker.get_effective_rate(110.) == (70 if enabled else 52)
    assert tracker.get_effective_rate(110.01) == 52
    tracker.reset()
    assert tracker.get_effective_rate(1000.) == 70


@pytest.mark.parametrize("enabled", [False, True])
def test_overlay_and_accounting_identical_origin(enabled: bool) -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from src.ojama_accounting import OjamaAccountingTracker
    from tests.test_exchange_event_overlay import Models, Signals, build_static, result
    overlay = ExchangeEventOverlay(Models(), build_static, Signals,
                                   margin_origin_first_placement=enabled)
    accounting = OjamaAccountingTracker(margin_origin_first_placement=enabled)
    accounting.reset(match_start_sec=10.)
    snapshot = SimpleNamespace(net_balance_capped=0., forecast_p1=0.,
                               total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                            chain_total_score_p1=None, chain_total_score_p2=None)
    for stamp, values in ((10., ()), (14., (1, 2)), (16., (3, 4))):
        observed = result(stamp)
        observed.p2.confirmed_board = board(values)
        overlay.update(observed, snapshot, final, stamp, 1)
        accounting.observe_margin((observed.p1, observed.p2), stamp, 1)
    assert overlay._margin_elapsed(108.) == accounting._elapsed(108.) == (94. if enabled else 98.)


def test_cli_default_off_and_explicit_on() -> None:
    import argparse
    from src.exchange_event_cli import parse_exchange_event_args
    assert not parse_exchange_event_args(argparse.ArgumentParser(), []).margin_origin_first_placement
    args = parse_exchange_event_args(argparse.ArgumentParser(), ["--margin-origin-first-placement"])
    assert args.margin_origin_first_placement


@pytest.mark.parametrize("enabled", [False, True])
def test_accounting_finalized_send_changes_at_decay_boundary(enabled: bool) -> None:
    from src.ojama_accounting import OjamaAccountingTracker
    from tests.test_ojama_accounting import _finalize_direct
    tracker = OjamaAccountingTracker(margin_origin_first_placement=enabled)
    tracker.reset(match_start_sec=10.)
    tracker.observe_margin((side(), side()), 10., 1)
    tracker.observe_margin((side((1, 2)), side()), 14., 1)
    _finalize_direct(tracker, "p1", 700, 108.)
    assert tracker._p1.total_generated == (10 if enabled else 13)
    assert tracker._p1.last_effective_rate == (70 if enabled else 52)


@pytest.mark.parametrize("enabled", [False, True])
def test_firing_uses_margin_but_preserves_trained_phase_clock(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.exchange_event_overlay as module
    from src.exchange_event_features import SIDE_COLUMNS
    from tests.test_exchange_event_overlay import Models, Signals, build_static, result
    monkeypatch.setattr(module, "prefire_side_features", lambda *args: np.zeros(len(SIDE_COLUMNS)))
    overlay = module.ExchangeEventOverlay(Models(), build_static, Signals,
                                         margin_origin_first_placement=enabled)
    snapshot = SimpleNamespace(net_balance_capped=0., forecast_p1=0.,
                               total_dropped_to_p1=0, total_dropped_to_p2=0)
    final = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                            chain_total_score_p1=None, chain_total_score_p2=None)
    for stamp, values in ((10., ()), (14., (1, 2))):
        observed = result(stamp)
        observed.p1.confirmed_board = board(values)
        overlay.update(observed, snapshot, final, stamp, 1)
    fire = result(1.)
    fire.p1.chain_event.trigger_sec = 108.
    overlay.update(fire, snapshot, final, 108., 1)
    assert overlay.tracker._score_elapsed == (94. if enabled else 98.)
    assert overlay.tracker.firing.static.elapsed_sec == 98.


def test_accounting_game_boundary_resets_shared_clock() -> None:
    from src.ojama_accounting import OjamaAccountingTracker
    tracker = OjamaAccountingTracker(margin_origin_first_placement=True)
    tracker.observe_margin((side(), side()), 1., 1)
    tracker.observe_margin((side((1, 2)), side()), 2., 1)
    tracker.observe_margin((side(), side()), 200., 2)
    assert tracker._margin_clock.origin is None
    tracker.observe_margin((side(), side((3, 4))), 204., 2)
    assert tracker._margin_clock.origin == 204.


@pytest.mark.parametrize("module", ["scripts.visualize_advantage_overlay", "scripts.replay_exchange_event_20260926"])
@pytest.mark.parametrize("enabled", [False, True])
def test_render_and_replay_cli_wiring(module: str, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_exchange_event_production import cli_options
    argv = ["--production-exchange-event"]
    if "replay" in module:
        argv += ["input.jsonl.gz", "--out", "unused"]
    if enabled:
        argv.append("--margin-origin-first-placement")
    options = cli_options(module, argv, monkeypatch)
    assert options["margin_origin_first_placement"] is enabled


@pytest.mark.parametrize("times", [(4., 5.), (5., 4.), (4., None), (None, 4.)])
def test_recorded_first_placement_times(times: tuple[float | None, float | None]) -> None:
    clock = MarginClock()
    clock.observe((side(), side()), 1., 1, (None, None))
    clock.observe((side(state=BoardState.CHAIN), side()), 6., 1, times)
    assert clock.origin == 4.


def test_explicit_missing_times_do_not_guess_from_board() -> None:
    clock = MarginClock()
    clock.observe((side(), side()), 1., 1, (None, None))
    clock.observe((side((1, 2)), side()), 2., 1, (None, None))
    assert clock.origin is None


def test_recorded_times_reset_and_reject_previous_match() -> None:
    clock = MarginClock()
    clock.observe((side(), side()), 1., 1, (None, None))
    clock.observe((side(), side()), 5., 1, (4., 5.))
    clock.observe((side(), side()), 100., 2, (4., 5.))
    assert clock.origin is None
    clock.observe((side(), side()), 105., 2, (104., 105.))
    assert clock.origin == 104.


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 11.])
def test_explicit_invalid_or_future_times_rejected(value: float) -> None:
    observation = PlacementObservation(10., BoardState.STABLE, None, None, value)
    assert first_placement_origin([observation]) is None


def test_pipeline_times_are_read_without_mutation() -> None:
    from src.margin_clock import placement_times_from_pipeline
    pipeline = SimpleNamespace(_first_move_sec_1p=5., _first_move_sec_2p=4.)
    assert placement_times_from_pipeline(pipeline) == (5., 4.)
    assert vars(pipeline) == dict(_first_move_sec_1p=5., _first_move_sec_2p=4.)


@pytest.mark.parametrize("times", [None, (None, None), (14., 15.)])
def test_recorded_times_round_trip(times: tuple | None, tmp_path: Path) -> None:
    from src.exchange_event_record import ExchangeEventRecorder, read_records
    from src.exchange_event_record import SNAPSHOT_FIELDS, FINALIZATION_FIELDS, FIRST_PLACEMENT_ARGUMENT_INDEX
    from tests.test_exchange_event_overlay import result
    path = tmp_path / "record.jsonl.gz"
    recorder = ExchangeEventRecorder(path, "test", False, Path("models/exchange_event_v3"))
    snapshot = SimpleNamespace(**dict.fromkeys(SNAPSHOT_FIELDS, 0))
    final = SimpleNamespace(**dict.fromkeys(FINALIZATION_FIELDS, 0))
    recorder.update(result(16.), snapshot, final, 16., 1, (None, None),
                    (None, None), (False, False), times)
    recorder.close()
    inputs = next(row["args"] for row in read_records(path) if row["kind"] == "update")
    if times is None:
        assert len(inputs) == FIRST_PLACEMENT_ARGUMENT_INDEX
    else:
        assert inputs[FIRST_PLACEMENT_ARGUMENT_INDEX] == times


def test_live_render_delivers_same_observed_origin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.visualize_advantage_overlay as vao
    from src.exchange_event_record import read_records, FIRST_PLACEMENT_ARGUMENT_INDEX
    from src.ojama_accounting import OjamaAccountingTracker
    from tests.test_exchange_event_overlay import stub, FPS, Pipeline, result
    from src.exchange_event_overlay import ExchangeEventOverlay
    stub(monkeypatch)
    observed_origins, overlay_origins = [], []
    original = OjamaAccountingTracker.observe_margin
    overlay_update = ExchangeEventOverlay.update

    def update(pipe: Pipeline, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        pipe._first_move_sec_1p = .3 if t_sec >= .3 else None
        pipe._first_move_sec_2p = .6 if t_sec >= .6 else None
        return result(t_sec)

    def observe(tracker: OjamaAccountingTracker, *args: object) -> None:
        original(tracker, *args)
        observed_origins.append(tracker._margin_clock.origin)

    def observe_overlay(overlay: ExchangeEventOverlay, *args: object) -> None:
        overlay_update(overlay, *args)
        overlay_origins.append(overlay._margin_clock.origin)

    monkeypatch.setattr(Pipeline, "update", update)
    monkeypatch.setattr(OjamaAccountingTracker, "observe_margin", observe)
    monkeypatch.setattr(ExchangeEventOverlay, "update", observe_overlay)
    path = tmp_path / "inputs.jsonl.gz"
    vao.generate(Path("short.mp4"), tmp_path / "overlay.mp4", 4, 1 / FPS,
        render=False, enable_exchange_event_update=True, margin_origin_first_placement=True,
        exchange_event_m0_predictor=lambda b, q: .5, exchange_event_record_path=path)
    assert .3 in observed_origins and set(observed_origins) <= {None, .3}
    assert overlay_origins == observed_origins
    recorded = [r["args"][FIRST_PLACEMENT_ARGUMENT_INDEX] for r in read_records(path) if r["kind"] == "update"]
    assert (.3, .6) in recorded


def test_new_functions_stay_within_fifty_lines() -> None:
    import ast
    root = Path(__file__).resolve().parents[1]
    for name in ("src/margin_clock.py", "scripts/audit_margin_origins.py",
                 "scripts/measure_margin_clock.py", "scripts/report_margin_clock.py",
                 "tests/test_margin_clock.py"):
        for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef):
                assert node.end_lineno - node.lineno + 1 <= 50, (name, node.name)
