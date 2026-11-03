"""E7の実表示EMA・v2物差しとE6bの到達／閉鎖条件を固定する。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import report_e7_exchange_20260927 as metrics
from scripts.e6b_exchange_reach_20260926 import gate_state
from scripts.visualize_advantage_overlay import (
    EMA_ALPHA, _ExchangeDisplayEMA, _exchange_display, _winprob_to_adv,
)
from src.exchange_event_tracker import (
    EXCHANGE_IDLE_TIMEOUT_SEC, S3_END_QUIET_SEC, ExchangeEventTracker,
)
from tests.test_exchange_event_tracker import fire, static, tracker

FRAME_SEC = 1 / 30


def display(values: list[float], games: list[int] | None = None) -> dict:
    """計測対象の実表示と、独立した未平滑化確率を用意する。"""
    return dict(t_sec=np.arange(len(values), dtype=float),
                display_adv=np.array(values, dtype=float),
                game_idx=np.array(games if games is not None else [1] * len(values)),
                display_p1=np.full(len(values), .8), source=np.full(len(values), "G_fe"))


@pytest.mark.parametrize("probability", [.01, .3, .5, .8, .99])
def test_ema_converts_before_smoothing_and_preserves_raw(probability: float) -> None:
    """確率を先に平滑化して非線形変換する誤りと、評価値の上書きを防ぐ。"""
    overlay = SimpleNamespace(tracker=SimpleNamespace(probability=probability))
    ema = _ExchangeDisplayEMA()
    raw = float(np.clip(_winprob_to_adv(probability), -100, 100))
    adv, smoothed = _exchange_display(overlay, 0, .5, ema, 0)
    assert adv == pytest.approx(EMA_ALPHA * raw)
    assert smoothed == pytest.approx(.5 + EMA_ALPHA * (probability - .5))
    assert overlay.tracker.probability == probability


def test_ema_same_frame_is_idempotent_and_advances_without_redraw() -> None:
    """描画を間引いても認識フレームごとに進み、再参照では進まない。"""
    ema = _ExchangeDisplayEMA()
    for frame in range(5):
        value = ema.apply(80, .8, frame * FRAME_SEC)
        assert ema.apply(-80, .2, frame * FRAME_SEC) == value
    assert value[0] == pytest.approx(80 * (1 - (1 - EMA_ALPHA) ** 5))


def test_missing_evaluation_uses_fallback_without_advancing_ema() -> None:
    overlay = SimpleNamespace(tracker=SimpleNamespace(probability=None))
    ema = _ExchangeDisplayEMA()
    assert _exchange_display(overlay, 17, .6, ema, 1) == (17, .6)
    assert (ema.adv, ema.probability, ema.last_sec) == (0, .5, None)


@pytest.mark.parametrize("values,games,expected", [
    ([3, 0, -3], None, 1), ([2.99, -2.99, 0], None, 0),
    ([3, 0, 3], None, 0), ([3, -3], [1, 2], 0),
    ([3, -3, 3], None, 2),
])
def test_m4_even_band_and_game_boundaries(values: list, games: list | None, expected: int) -> None:
    result = metrics.m4(display(values, games))
    assert result["flips"] == expected
    assert result["minutes"] == len(values) / metrics.e3.FPS / metrics.e3.SECONDS_PER_MINUTE


def test_m4_rejects_missing_display() -> None:
    with pytest.raises(ValueError):
        metrics.m4(display([0, float("nan")]))


@pytest.mark.parametrize("values,target,seconds", [
    ([0, 2, 10, 15], 20, 0), ([0, 2, 9, 10], 20, 1),
    ([0, -2, -9, -10], -20, 1), ([0, 2, 9, 9], 20, float("inf")),
])
def test_m2_direction_midpoint_and_unreached(values: list, target: float, seconds: float) -> None:
    result = metrics.delay(display(values), 1, 1, 2, 4, target)
    assert result["base_sec"] == 0
    assert result["seconds"] == seconds


@pytest.mark.parametrize("change,excluded", [(4.99, True), (5, False), (-5, False)])
def test_m2_small_change_boundary(change: float, excluded: bool) -> None:
    result = metrics.delay(display([0, change, change]), 1, 1, 1, 3, change)
    assert (result.get("excluded") == "small_change") is excluded


def test_m2_cutoff_and_match_do_not_supply_future_reach() -> None:
    data = display([0, 0, 0, 20], [1, 1, 1, 2])
    assert np.isinf(metrics.delay(data, 1, 1, 2, 4, 20)["seconds"])
    data["game_idx"][:] = 1
    assert np.isinf(metrics.delay(data, 1, 1, 2, 3, 20)["seconds"])
    assert metrics.delay(data, 1, 0, 2, 4, 20) == dict(excluded="missing_prefire")


def event() -> dict:
    """物理終了・閉鎖・評価更新が別フレームの撃ち合い。"""
    return dict(exchange_id=1, game_idx=1, trigger_sec=1, closed_sec=3,
                close_reason="confirmed_after_score", chains=[dict(end_signal_sec=2)])


def test_m2_target_uses_first_postclose_raw_gfe_before_next_exchange() -> None:
    data, first = display([0] * 7), event()
    data["source"][3] = "S3"
    following = dict(exchange_id=2, game_idx=1, trigger_sec=5)
    target = metrics.event_target(first, [first, following], data)
    assert target == dict(end_sec=2, cutoff_sec=5,
                          target=_winprob_to_adv(.8), target_sec=4)
    following["trigger_sec"] = 4
    assert metrics.event_target(first, [first, following], data) == dict(
        excluded="missing_postclose_static")


@pytest.mark.parametrize("field,value,reason", [
    ("chains", [dict(end_signal_sec=None)], "missing_end"),
    ("closed_sec", None, "not_completed"),
    ("close_reason", "match_boundary", "not_completed"),
])
def test_m2_common_exclusions(field: str, value: object, reason: str) -> None:
    first = event()
    first[field] = value
    assert metrics.event_target(first, [first], display([0] * 7)) == dict(excluded=reason)


def test_m2_unreached_remains_in_median_denominator() -> None:
    rows = [dict(modes={mode: dict(seconds=value) for mode in metrics.MODES})
            for value in (0, float("inf"), float("inf"))]
    rows.append(dict(excluded="missing_end"))
    result = metrics.m2_summary(rows)
    assert result["exchanges"] == 4 and result["common_excluded"] == {"missing_end": 1}
    assert result["on"]["eligible"] == 3 and result["on"]["unreached"] == 2
    assert np.isinf(result["on"]["median_seconds"])


def test_s3_gets_one_display_frame_before_confirmed_close(tracker: ExchangeEventTracker) -> None:
    """確定盤面が揃ってもS3生成フレーム内のG_fe直行を禁止する。"""
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 3, 700)
    ready = 3 + S3_END_QUIET_SEC
    assert gate_state(tracker, ready - FRAME_SEC) == "end_quiet"
    assert not tracker.static(static(), ready - FRAME_SEC, (ready, ready))
    tracker.finish_frame(ready)
    assert tracker.source == "S3"
    assert not tracker.static(static(), ready, (ready, ready))
    assert not tracker.static(static(), ready + FRAME_SEC, (3, ready))
    assert tracker.static(static(), ready + FRAME_SEC, (ready, ready))
    assert tracker.records[0].close_reason == "confirmed_after_score"
    assert [row["source"] for row in tracker.records[0].values] == ["S1", "S3", "G_fe"]


@pytest.mark.parametrize("ended,scored,gate", [
    (False, False, "missing_end"), (False, True, "missing_end"),
    (True, False, "missing_score_ready"),
])
def test_unready_exchange_never_closes_via_static(
    tracker: ExchangeEventTracker, ended: bool, scored: bool, gate: str,
) -> None:
    fire(tracker)
    if ended:
        tracker.end("1P", 3, "next")
    if scored:
        tracker.finalize("1P", 3, 700)
    tracker.finish_frame(4)
    assert gate_state(tracker, 4) == gate
    assert not tracker.close_confirmed(4, (4, 4))
    assert tracker.current is not None


def test_timeout_closes_unreached_exchange_without_fabricating_s3(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    deadline = 2 + EXCHANGE_IDLE_TIMEOUT_SEC
    tracker.finish_frame(deadline - FRAME_SEC)
    assert tracker.current is not None
    tracker.finish_frame(deadline)
    assert tracker.current is None and tracker.source == "G_fe"
    record = tracker.records[0]
    assert record.close_reason == "activity_timeout" and record.closed_sec == deadline
    assert [row["source"] for row in record.values] == ["S1"]
