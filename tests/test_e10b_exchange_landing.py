"""確定S3、着地までの保持、新しい応手による再判定を検証する。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src import exchange_event_landing as landing
from src.board import Board
from src.exchange_event_overlay import ConfirmedSide, ExchangeEventOverlay
from tests.test_e10_exchange_landing import board_at_height
from tests.test_exchange_event_tracker import fire, static, tracker


def setup_projection(tracker: object) -> tuple:
    """2Pに20個を送る、窒息高さを丸1段超える盤面を用意する。"""
    fire(tracker)
    tracker.observe_score("1P", 2, None, formula_total=1400)
    tracker.finish_frame(2)
    snapshot = SimpleNamespace(total_dropped_to_p1=0, total_dropped_to_p2=0)
    histories = [[ConfirmedSide(1, b, np.ones(4, dtype=int))]
                 for b in (Board(), board_at_height(11))]
    overlay = SimpleNamespace(tracker=tracker, _start=0, _m0=lambda *args: .5,
        _build_static=lambda *args: static(), _history=histories, _snapshots=[(1, snapshot)])
    result = SimpleNamespace(p1=SimpleNamespace(chain_event=None),
                             p2=SimpleNamespace(chain_event=None))
    projection = landing.ExchangeLandingProjection()
    overlay._landing_projection = projection
    return projection, overlay, result, snapshot


@pytest.mark.parametrize("available,source", [(0, "unavoidable_death"), (100, "S3_landing")])
def test_confirmed_s3_uses_same_projection(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                          available: int, source: str) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    tracker.end("1P", 2.1, "next")
    tracker.finalize("1P", 2.2, 1400)
    tracker.finish_frame(2.5)
    assert tracker.source == "S3"
    monkeypatch.setattr(landing, "future_send", lambda *args: available)
    projection.update(overlay, result, snapshot, 2.5)
    assert tracker.source == source
    assert tracker.probability == pytest.approx(.98 if available == 0
                                               else landing.logit_mean(.8, .4))


@pytest.mark.parametrize("change", ["score", "s3", "dropped", "timeout"])
def test_death_survives_nonresponse_changes(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                           change: str) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    monkeypatch.setattr(landing, "future_send", lambda *args: 1000)
    if change == "score":
        tracker.observe_score("1P", 2.1, None, formula_total=2100)
        tracker.finish_frame(2.1)
    elif change == "s3":
        tracker.current.values.append(dict(source="S3", p1=.49, t_sec=2.1))
    elif change == "dropped":
        snapshot.total_dropped_to_p2 = 20
    else:
        tracker.finish_frame(6)
        assert tracker.current is None
    projection.update(overlay, result, snapshot, 6 if change == "timeout" else 2.1)
    assert tracker.source == "unavoidable_death" and tracker.probability == .98
    # 静止評価による上書きも着地までは防ぐ。
    ExchangeEventOverlay._static(overlay, snapshot, 6)
    assert tracker.probability == .98


def test_new_receiver_chain_immediately_reassesses(tracker: object,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    fire(tracker, 2.1, (None, 2.1))
    tracker.finish_frame(2.1)
    monkeypatch.setattr(landing, "future_send", lambda *args: 1000)
    projection.update(overlay, result, snapshot, 2.1)
    assert projection.death is None
    assert tracker.source == "S3_landing" and tracker.probability != .98


def test_release_requires_landing_and_both_confirmed_boards(tracker: object,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    tracker.end("1P", 2.1, "next")
    tracker.finalize("1P", 2.2, 700)
    tracker.finish_frame(2.5)
    projection.update(overlay, result, snapshot, 2.5)
    overlay._history[0].append(ConfirmedSide(2.6, Board(), np.ones(4, dtype=int)))
    overlay._history[1].append(ConfirmedSide(2.6, board_at_height(11), np.ones(4, dtype=int)))
    projection.update(overlay, result, snapshot, 2.6)
    assert projection.death is not None  # STABLEだけでは着地と扱わない。
    tracker.fall_start("2P", 2.7)
    tracker.landing("2P", 2.8)
    overlay._history[1].append(ConfirmedSide(2.8, board_at_height(11), np.ones(4, dtype=int)))
    projection.update(overlay, result, snapshot, 2.8)
    assert projection.death is not None
    overlay._history[0].append(ConfirmedSide(2.9, Board(), np.ones(4, dtype=int)))
    overlay._history[1].append(ConfirmedSide(2.9, Board(), np.ones(4, dtype=int)))
    projection.update(overlay, result, snapshot, 2.9)
    assert projection.death is None
    ExchangeEventOverlay._static(overlay, snapshot, 2.9)
    assert tracker.source == "G_fe" and tracker.probability == .4


def test_boundary_clears_held_death(tracker: object, monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    tracker.boundary(2, 3)
    projection.update(overlay, result, snapshot, 3)
    assert projection.death is None and tracker.probability is None


def test_existing_response_steps_reassess_reduced_attack(tracker: object,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    fire(tracker, 2.1, (None, 2.1))
    tracker.finish_frame(2.1)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2.1)
    response_id = projection.response_id
    fire(tracker, 2.2, (None, 2.2))
    tracker.observe_score("2P", 2.2, None, formula_total=140)
    tracker.finish_frame(2.2)
    monkeypatch.setattr(landing, "future_send", lambda *args: 1000)
    projection.update(overlay, result, snapshot, 2.2)
    assert projection.response_id is None and response_id is not None
    assert tracker.source == "S3_landing" and tracker.probability != .98


def test_confirmed_s3_does_not_reland_dropped_ojama(tracker: object,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    tracker.current.values.append(dict(source="S3", p1=.8, t_sec=2.1))
    snapshot.total_dropped_to_p2 = 20
    # 基準スナップショットは着弾前のまま保持する。
    overlay._snapshots = [(1, SimpleNamespace(total_dropped_to_p1=0, total_dropped_to_p2=0))]
    def unexpected(*args: object) -> None:
        pytest.fail("既着地分を再投下している")
    monkeypatch.setattr(projection, "_evaluate", unexpected)
    projection.update(overlay, result, snapshot, 2.1)
    assert tracker.source == "S3" and tracker.probability == .8


def test_confirmed_composition_always_uses_original_s3(tracker: object,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    tracker.current.values.append(dict(source="S3", p1=.8, t_sec=2.1))
    monkeypatch.setattr(landing, "future_send", lambda *args: 1000)
    projection.update(overlay, result, snapshot, 2.1)
    first = tracker.probability
    snapshot.total_dropped_to_p2 = 1
    projection.update(overlay, result, snapshot, 2.2)
    assert tracker.probability == first == pytest.approx(landing.logit_mean(.8, .4))


def test_landing_before_score_confirmation_can_release(tracker: object,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    tracker.end("1P", 2.1, "next")
    tracker.landing("2P", 2.2)
    tracker.finalize("1P", 2.3, 700)
    tracker.finish_frame(2.5)
    projection.update(overlay, result, snapshot, 2.5)
    for history in overlay._history:
        history.append(ConfirmedSide(2.6, Board(), np.ones(4, dtype=int)))
    projection.update(overlay, result, snapshot, 2.6)
    assert projection.death is None
    ExchangeEventOverlay._static(overlay, snapshot, 2.6)
    assert tracker.source == "G_fe"
