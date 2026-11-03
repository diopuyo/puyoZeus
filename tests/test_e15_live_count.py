"""E15の鮮度、連鎖消費、既定OFF互換性を検証する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src import exchange_event_count_features as counts
from src.board import Board
from src.exchange_event_overlay import ConfirmedSide, ExchangeEventOverlay
from src.exchange_event_evaluator import count_sides
from tests.test_e14_count_features import ROWS, inputs
from scripts import train_exchange_event_models_v2_20260927 as v2


def overlay(live: bool = True) -> ExchangeEventOverlay:
    """実盤面fixtureと追従対象のtrackerを用意する。"""
    _, firing, _ = inputs(ROWS[0])
    value = ExchangeEventOverlay(SimpleNamespace(count_features=True), None, None, live_count=live)
    value._start = 0.
    value.tracker.current = SimpleNamespace(exchange_id=1, chains=[])
    value.tracker.firing = firing
    value.tracker._score_elapsed = 0.
    value._history = [[ConfirmedSide(1., Board.from_list(grid.tolist()), queue)]
        for grid, queue in zip(firing.count_observation.grids, firing.count_observation.queues)]
    return value


def test_update_changes_counts_and_invalidates_s3() -> None:
    value = overlay()
    value._refresh_features(None, 2.)
    before = count_sides(value.tracker.firing)
    value._history[0].append(ConfirmedSide(3., Board(), np.zeros(4)))
    value.tracker._s3_sec, value.tracker._provisional_key = 2., (1,)
    value._refresh_features(None, 3.)
    after = count_sides(value.tracker.firing)
    assert not np.array_equal(before[0], after[0])
    assert value.tracker._s3_sec is None and value.tracker._provisional_key is None
    event = value.tracker.firing
    value._refresh_features(None, 3.1)
    assert value.tracker.firing is event


def test_chain_projection_then_actual_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    value = overlay()
    final = np.zeros((13, 6), dtype=np.int8)
    chain = SimpleNamespace(end_signal_sec=None, predicted_final_board=final.tolist(), chain_id=1)
    monkeypatch.setattr(value.tracker, "latest_chain", lambda side: chain if side == "1P" else None)
    value._refresh_features(None, 2.)
    assert not value.tracker.firing.count_observation.grids[0].any()
    # 発火フラグがTrueでも完走済み盤面を再消費しない。
    actual = count_sides(value.tracker.firing)
    observation = value.tracker.firing.count_observation
    np.testing.assert_array_equal(actual, counts.side_features(observation, (False, False)))
    chain.end_signal_sec = 3.
    value._history[0].append(ConfirmedSide(4., Board.from_list(ROWS[0]["pre"][0]["grid"]), np.zeros(4)))
    value._refresh_features(None, 4.)
    assert value.tracker.firing.count_observation.grids[0].any()


def test_off_preserves_input_bytes() -> None:
    value = overlay(False)
    event = value.tracker.firing
    before = count_sides(event).tobytes()
    value._history[0].append(ConfirmedSide(3., Board(), np.zeros(4)))
    value._refresh_features(None, 3.)
    assert value.tracker.firing is event
    assert count_sides(value.tracker.firing).tobytes() == before


def test_cache_uses_effective_rate() -> None:
    grid = np.zeros((13, 6), dtype=np.int8).tobytes()
    counts._fire_at_rate.cache_clear()
    first = counts.fire(grid, (1, 2, 3, 4), 1., (1,))
    second = counts.fire(grid, (1, 2, 3, 4), 2., (1,))
    assert first is second
    assert counts._fire_at_rate.cache_info().misses == 1


@pytest.mark.parametrize("elapsed", [0., 95.5, 96., 96.01, 111.99, 112., 320., 1000.])
def test_rate_cache_preserves_values(elapsed: float) -> None:
    """マージン境界を含めて、旧学習側の生時刻計算と全K値が一致する。"""
    raw, firing, _ = inputs(ROWS[0])
    grid = firing.count_observation.grids[0].tobytes()
    queue = tuple(int(x) for x in firing.count_observation.queues[0])
    actual = counts.fire(grid, queue, elapsed, v2.iv.NEAR_FUTURE_K_LEVELS)
    expected = v2.fire(grid, queue, elapsed, v2.iv.NEAR_FUTURE_K_LEVELS)
    assert actual.tobytes() == expected.tobytes()


def test_e3_flag_selects_v3_only_when_enabled() -> None:
    """E3親CLIでもONが無効なv1へ配線されない。OFFコマンドは不変。"""
    from pathlib import Path
    from scripts.run_e3_exchange_eval_20260926 import command, SOURCES
    old = command(SOURCES[0], "on", Path("test"), 2)
    assert old == command(SOURCES[0], "on", Path("test"), 2, False)
    enabled = command(SOURCES[0], "on", Path("test"), 2, True)
    assert "--exchange-event-live-count" in enabled
    assert enabled[enabled.index("--exchange-event-model-dir")+1] == "models/exchange_event_v3"
