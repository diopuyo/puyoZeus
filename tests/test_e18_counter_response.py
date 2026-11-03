"""応手量の保存則、選択境界、予測層への限定と既定OFFを検証する。"""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS
import numpy as np
import pytest
from src import exchange_event_landing as landing
from src.board import Board
from src.exchange_event_overlay import ConfirmedSide, ExchangeEventOverlay
from tests.test_e10b_exchange_landing import setup_projection
from tests.test_exchange_event_tracker import tracker
from tests.test_e11_review_data_panel import fixture_overlay, review_row


@pytest.mark.parametrize("receiver", [0, 1])
@pytest.mark.parametrize("send,net,selected", [(0, 20, False), (19, 1, False),
    (19.9, 1, False), (20, 0, True), (21, -1, True), (100, -80, True)])
def test_counter_conservation_and_selection(monkeypatch: pytest.MonkeyPatch,
        receiver: int, send: float, net: int, selected: bool) -> None:
    projection = landing.ExchangeLandingProjection(counter_response=True)
    latest = tuple(ConfirmedSide(1+i, Board(), np.ones(4, int)) for i in range(2))
    overlay = NS(tracker=NS(_score_elapsed=10, latest_chain=lambda *a: None))
    incoming = [0, 0]
    incoming[receiver] = 20
    seen = []
    monkeypatch.setattr(projection, "_receivers", lambda *a: (tuple(s.board for s in latest), [0, 0]))
    monkeypatch.setattr(projection, "_chaining", lambda *a: False)
    monkeypatch.setattr(landing, "future_send", lambda *a: send)
    monkeypatch.setattr(projection, "_landing_gfe", lambda o, s, l, n, t: seen.append(n) or .7)
    value = projection._counter_projection(overlay, None, latest, incoming, (1, 2), .3, 10.)
    assert value["response_incoming"][receiver] == max(0, net)
    assert value["response_incoming"][1-receiver] == max(0, -net)
    assert value["response_surplus"][receiver] == max(0, -net)
    assert value["response_selected"] is selected
    assert value["gfe_no_response_p1"] == .3
    assert value["gfe_response_p1"] == (.3 if send == 0 else .7)
    assert value["response_layer"] == "prediction" and incoming[receiver] == 20


def test_off_does_not_search_or_add_dto() -> None:
    projection = landing.ExchangeLandingProjection()
    assert projection._counter_projection(None, None, (), [], (), .3, 0.) == {}
    overlay = ExchangeEventOverlay(NS(), None, None)
    assert not overlay._landing_projection.counter_response


def test_counter_flag_does_not_enable_rejected_layers() -> None:
    overlay = ExchangeEventOverlay(NS(), None, None, live_count=True,
                                   death_guard=True, landing_counter_response=True)
    assert overlay._landing_projection.counter_response
    assert overlay._e16.death_enabled
    assert not overlay._e16.sync_enabled and not overlay._e16.layer_enabled


@pytest.mark.parametrize("send,selected", [(10, False), (20, True), (100, True)])
def test_selection_reaches_probability(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                        send: int, selected: bool) -> None:
    projection, overlay, observed, snapshot = setup_projection(tracker)
    projection.counter_response = True
    monkeypatch.setattr(landing, "future_send", lambda *a: send)
    monkeypatch.setattr(projection, "_verified_attack", lambda *a: False)
    monkeypatch.setattr(projection, "_landing_gfe", lambda o, s, l, n, t: .3 if n == [0, 20] else .7)
    projection.update(overlay, observed, snapshot, 2.)
    value = projection.last
    assert value["response_selected"] is selected
    assert value["gfe_p1"] == (.7 if selected else .3)
    assert tracker.probability == landing.logit_mean(value["base_p1"], value["gfe_p1"])


def test_remaining_hands_and_confirmed_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    projection = landing.ExchangeLandingProjection(True)
    latest = tuple(ConfirmedSide(1., Board(), np.array([1, 2, 3, 4])) for _ in range(2))
    latest[1].board._grid[-1, 0] = 2
    calls = []
    monkeypatch.setattr(projection, "_receivers", lambda *a: (tuple(s.board for s in latest), [0, 0]))
    monkeypatch.setattr(projection, "_chaining", lambda *a: False)
    monkeypatch.setattr(projection, "_landing_gfe", lambda *a: .6)
    monkeypatch.setattr(landing, "future_send", lambda *a: calls.append(a) or 30)
    projection._counter_projection(NS(tracker=NS(_score_elapsed=9, latest_chain=lambda *a: None)),
                                  None, latest, [0, 20], (1, 3), .4, 2.)
    assert calls[0][0] == latest[1].board._grid.tobytes()
    assert calls[0][3:] == ((1, 2, 3, 4), 3, 9)


def test_unresolved_active_chain_is_not_additional_response(monkeypatch: pytest.MonkeyPatch) -> None:
    projection = landing.ExchangeLandingProjection(True)
    latest = tuple(ConfirmedSide(1., Board(), np.ones(4, int)) for _ in range(2))
    monkeypatch.setattr(projection, "_receivers", lambda *a: (tuple(s.board for s in latest), [0, 999]))
    monkeypatch.setattr(projection, "_chaining", lambda *a: True)
    monkeypatch.setattr(projection, "_completion_board", lambda *a: None)
    value = projection._counter_projection(NS(tracker=NS(latest_chain=lambda *a: None)),
                                          None, latest, [0, 20], (1, 3), .4, 2.)
    assert value["response_send"] == [None, None]
    assert not value["response_selected"] and value["response_incoming"] == [0, 20]


def test_dto_to_csv_keeps_both_predictions_and_current() -> None:
    overlay = fixture_overlay()
    overlay._landing_projection.last.update(gfe_no_response_p1=.9, gfe_response_p1=.6,
        response_selected=True, response_layer="prediction", response_send=[None, 12],
        response_incoming=[2, 0], response_surplus=[0, 2], response_board_sec=[1., 1.5])
    row = review_row(overlay)
    assert row["p1_landing_no_response"] == .9 and row["p1_landing_response"] == .6
    assert row["p1_G_fe"] == .55
    assert row["2P_response_send"] == 12 and row["1P_response_incoming"] == 2
    assert row["landing_response_layer"] == "prediction"


@pytest.mark.parametrize("active", [True, False])
def test_already_counted_fire_is_not_added_again(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    projection = landing.ExchangeLandingProjection(True)
    latest = tuple(ConfirmedSide(1., Board(), np.ones(4, int)) for _ in range(2))
    completed = Board()
    completed._grid[-1, 0] = 2
    calls = []
    monkeypatch.setattr(projection, "_receivers", lambda *a: (tuple(s.board for s in latest), [0, 999]))
    monkeypatch.setattr(projection, "_chaining", lambda *a: active)
    monkeypatch.setattr(projection, "_completion_board", lambda *a: completed)
    monkeypatch.setattr(projection, "_landing_gfe", lambda *a: .6)
    monkeypatch.setattr(landing, "future_send", lambda *a: calls.append(a) or 10)
    overlay = NS(tracker=NS(_score_elapsed=9, latest_chain=lambda *a: NS(end_signal_sec=2.)))
    value = projection._counter_projection(overlay, None, latest, [0, 20], (1, 3), .4, 2.)
    assert value["response_send"] == [None, 10] and value["response_incoming"] == [0, 10]
    assert calls[0][0] == completed._grid.tobytes()


def test_render_and_replay_defaults_off() -> None:
    from scripts.visualize_advantage_overlay import generate
    from scripts.replay_exchange_event_20260926 import replay
    for function in (generate, replay):
        assert inspect.signature(function).parameters["landing_counter_response"].default is False
    assert inspect.signature(generate).parameters["exchange_event_death_guard"].default is False
