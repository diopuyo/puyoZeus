"""打ち返し・段別得点・確定後の対応連鎖で最新評価を保持する。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.visualize_advantage_overlay import EMA_ALPHA, _ExchangeDisplayEMA, _exchange_display
from src.exchange_event_evaluator import StaticInput
from src.exchange_event_features import SIDE_COLUMNS, score_features
from src.exchange_event_tracker import ExchangeEventTracker
from tests.test_exchange_event_tracker import fire, static, tracker


def test_reply_uses_current_inputs_and_flags(tracker: ExchangeEventTracker) -> None:
    """古い発火盤面で再計算しただけの更新を検出する。"""
    fire(tracker)
    current = StaticInput(np.full_like(static().d_features, .2), .5, 3)
    sides = np.full((2, len(SIDE_COLUMNS)), .3)
    tracker.fire(t_sec=3, triggers=(None, 3), static=current,
                 prefire_sides=sides, score_elapsed_sec=3)
    assert tracker.firing.static is current
    np.testing.assert_array_equal(tracker.firing.prefire_sides, sides)
    assert tracker.firing.firing == (True, True)
    assert [v["t_sec"] for v in tracker.current.values] == [2, 3]


@pytest.mark.parametrize("elapsed", [2, 150, 250])
def test_each_step_uses_existing_score_conversion(tracker: ExchangeEventTracker, elapsed: int) -> None:
    """段重複は一度だけ、相殺量とマージンレートは既存式と一致する。"""
    captured = []
    tracker.models.predict_source_probability = lambda name, x: (captured.append((name, x)), .4)[1]
    fire(tracker, elapsed, (elapsed, elapsed))
    for step, total in enumerate((40, 360, 1000)):
        t = elapsed + step / 2
        tracker.observe_score("1P", t, None, formula_total=total)
        tracker.observe_score("2P", t, None, formula_total=2960)
        tracker.finish_frame(t)
        assert tracker.source == "S3_provisional" and tracker._s3_sec is None
        np.testing.assert_allclose(captured[-1][1][-7:],
            score_features(np.zeros(2), np.array([total, 2960]), elapsed))
        count = len(captured)
        tracker.finish_frame(t + 1 / 30)
        assert len(captured) == count
    assert len(tracker.current.values) == 4


def test_response_after_s3_preserves_completed_scores(tracker: ExchangeEventTracker) -> None:
    """qの1000対2960の後に40点が加わっても、古いS1を表示しない。"""
    fire(tracker, triggers=(2, 2))
    for side, total in (("1P", 1000), ("2P", 2960)):
        tracker.end(side, 3, "next")
        tracker.finalize(side, 3, total)
    tracker.finish_frame(3 + 10 / 30)
    assert tracker.source == "S3"
    fire(tracker, 4, (4, None))
    assert tracker.source == "S3_provisional"
    assert tracker.current.values[-1]["score_totals"] == [1000, 2960]
    tracker.observe_score("1P", 4, None, formula_total=40)
    tracker.finish_frame(4)
    assert tracker.current.values[-1]["score_totals"] == [1040, 2960]
    assert [c.score_delta for c in tracker.current.chains[:2]] == [1000, 2960]
    tracker.end("1P", 5, "next")
    tracker.finalize("1P", 5, 40)
    tracker.finish_frame(5 + 10 / 30)
    assert tracker.source == "S3"


@pytest.mark.parametrize("reason", ["missing", "activity", "decrease"])
def test_s3_never_reverts_to_saved_s1(tracker: ExchangeEventTracker, reason: str) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 1100, 100, 1000)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 3, 1000)
    tracker.finish_frame(3 + 10 / 30)
    old = tracker.probability
    if reason == "missing":
        tracker.missing_input("missing_display_score_or_baseline", 4, "S3")
    elif reason == "activity":
        tracker.activity("1P", 4)
    else:
        tracker.observe_score("1P", 4, 1000)
    assert tracker.probability == old and tracker.source == "S3"


def test_held_provisional_value_converges_every_frame(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, None, formula_total=1000)
    tracker.finish_frame(2)
    ema = _ExchangeDisplayEMA()
    overlay = SimpleNamespace(tracker=tracker)
    for frame in range(30):
        _, value = _exchange_display(overlay, 0, .5, ema, 2 + frame / 30)
    assert value == pytest.approx(.8 + (.5 - .8) * (1 - EMA_ALPHA) ** 30)
    assert len(tracker.current.values) == 2


def test_boundary_clears_provisional_scores(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, None, formula_total=1000)
    tracker.finish_frame(2)
    tracker.boundary(2, 3)
    fire(tracker, 4, (4, None))
    tracker.finish_frame(4)
    assert tracker.source == "S1" and tracker._provisional_key is None
