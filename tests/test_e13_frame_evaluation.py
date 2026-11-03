"""入力確定前の評価禁止と、段間の終了候補による予測消失を検証する。"""
from __future__ import annotations

import numpy as np
import pytest

from src.exchange_event_tracker import ExchangeEventTracker, S3_END_QUIET_SEC
from tests.test_exchange_event_tracker import fire, tracker
from scripts.exchange_value_spikes import value_spikes
from scripts.review_data_panel import expand_review_graph
from scripts.visualize_advantage_overlay import _font
from tests.test_e12b_exchange_completion import active_projection
from src import exchange_event_landing as landing
from types import SimpleNamespace


def test_model_waits_for_both_scores_and_prediction(tracker: ExchangeEventTracker) -> None:
    """発火を先に通知しても、完走予測投入後にモデルを一度だけ呼ぶ。"""
    calls = []
    tracker.models.predict_source_probability = lambda name, x: (calls.append(name), .8)[1]
    tracker.begin_frame()
    fire(tracker, triggers=(2, 2))
    for side, score in [('1P', 78150), ('2P', 28960)]:
        tracker.latest_chain(side).predicted_final_score = score
        tracker.observe_score(side, 2, None, formula_total=100)
    assert calls == []
    tracker.confirm_frame_inputs(2)
    tracker.finish_frame(2)
    assert len(calls) == 1
    assert tracker.current.values[-1]['score_totals'] == [78150, 28960]
    assert len(tracker.current.values) == 1


@pytest.mark.parametrize('delay', [0, .1, .3])
def test_unconfirmed_end_keeps_completion(tracker: ExchangeEventTracker, delay: float) -> None:
    fire(tracker)
    chain = tracker.latest_chain('1P')
    chain.predicted_final_score = 78150
    tracker.begin_frame()
    tracker.end('1P', 3, 'ojama')
    tracker.finalize('1P', 3, 100)
    tracker.confirm_frame_inputs(3 + delay)
    tracker.finish_frame(3 + delay)
    assert chain.provisional_score == 78150
    assert tracker.current.values[-1]['score_totals'] == [78150, 0]


def test_confirmed_end_uses_actual_score(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    chain = tracker.latest_chain('1P')
    chain.predicted_final_score = 78150
    tracker.end('1P', 3, 'next')
    tracker.finalize('1P', 3, 100)
    calls = []
    tracker.models.predict_source_probability = lambda name, x: (calls.append(name), .8)[1]
    tracker.confirm_frame_inputs(3 + S3_END_QUIET_SEC)
    tracker.finish_frame(3 + S3_END_QUIET_SEC)
    assert chain.provisional_score == 100
    assert tracker.source == 'S3'
    assert len(calls) == 1


@pytest.mark.parametrize('times,values,games,count', [
    ([0, .1, .6], [.8, .65, .75], [1, 1, 1], 1),
    ([0, .1, .601], [.8, .65, .8], [1, 1, 1], 0),
    ([0, .1, .2], [.8, .651, .8], [1, 1, 1], 0),
    ([0, .1, .2], [.8, .6, .749], [1, 1, 1], 0),
    ([0, .1, .2], [.8, .6, .8], [1, 1, 2], 0),
    ([0, .1, .2, .3, .4], [.8, .6, .8, .6, .8], [1] * 5, 2),
    ([0, .1, .2], [.2, .8, .2], [1] * 3, 1),
    ([0, .1, .2], [.2, .8, .9], [1] * 3, 0),
    ([0, .1, .2], [.2, np.nan, .2], [1] * 3, 0),
    ([], [], [], 0),
])
def test_spike_definition(times: list, values: list, games: list, count: int) -> None:
    result = value_spikes(np.array(times), np.array(values), np.array(games))
    assert result['count'] == count
    if count:
        assert result['per_minute'] == pytest.approx(count * 60 / (times[-1]-times[0]))


def test_review_graph_doubles_band_and_preserves_video() -> None:
    frame = np.full((450, 1280, 3), 71, dtype=np.uint8)
    drawn = expand_review_graph(frame, (0, 300, 1280, 150),
                               [(0, 100), (1, -100), (2, 50)], 2, 10, _font)
    assert drawn.shape == (600, 1280, 3)
    np.testing.assert_array_equal(frame[:300], drawn[:300])
    for color in [(130, 88, 45), (50, 58, 125)]:
        assert np.any(np.all(drawn[300:] == color, axis=2))


def test_current_step_is_seen_before_death_refresh(tracker: ExchangeEventTracker,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, observed, snapshot, chain = active_projection(tracker)
    monkeypatch.setattr(landing, 'future_send', lambda *args: 0)
    projection.update(overlay, observed, snapshot, 2.1)
    assert projection.death is not None
    observed.p2.chain_event = SimpleNamespace(chain_count=chain.predicted_chain_count + 1)
    projection.update(overlay, observed, snapshot, 2.2)
    assert projection.death is None
    assert tracker.source != 'unavoidable_death'
