"""E4: 静止更新・終了上限・終了撤回の回帰検証。"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.board_state_machine import BoardState
from src.exchange_event_overlay import ExchangeEventOverlay, formula_visible_from_pipeline
from src.exchange_event_tracker import EXCHANGE_IDLE_TIMEOUT_SEC, S3_END_QUIET_SEC
from tests.test_exchange_event_tracker import Models, fire, static, tracker
from tests.test_exchange_event_overlay import Signals, build_static, result


@pytest.mark.parametrize("finalized", [False, True])
def test_idle_timeout_releases_s1_and_s3(tracker: object, finalized: bool) -> None:
    fire(tracker)
    if finalized:
        tracker.end("1P", 2.1, "next")
        tracker.finalize("1P", 2.2, 700)
        tracker.finish_frame(2.5)
    last = 2.2 if finalized else 2
    tracker.finish_frame(last + EXCHANGE_IDLE_TIMEOUT_SEC - .01)
    assert tracker.current is not None
    tracker.finish_frame(last + EXCHANGE_IDLE_TIMEOUT_SEC)
    assert tracker.current is None and tracker.source == "G_fe"
    assert tracker.records[0].close_reason == "activity_timeout"


def test_score_activity_extends_timeout_and_revokes_end(tracker: object) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 100, 100)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3.1, 800, 100, 700)
    chain = tracker.latest_chain("1P")
    assert chain.end_signal_sec is None
    assert chain.end_signals[0]["revoked_sec"] == 3.1
    tracker.finish_frame(5)
    assert tracker.current is not None
    tracker.end("1P", 5, "next")
    tracker.observe_score("1P", 5, 800, 100, 700)
    tracker.finish_frame(5 + S3_END_QUIET_SEC)
    assert tracker.source == "S3"


def test_formula_resumption_blocks_s3_inside_grace(tracker: object) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 3, 700)
    tracker.finish_frame(3 + 9 / 30)
    assert tracker.source == "S1"
    tracker.activity("1P", 3 + 9 / 30)
    tracker.finish_frame(3 + 10 / 30)
    assert tracker.source == "S1"
    assert tracker.latest_chain("1P").end_signal_sec is None


def test_score_confirmed_pair_closes_without_landing(tracker: object) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 4, 700)
    tracker.finish_frame(4)
    assert not tracker.static(static(), 4.1, (4.1, 3.9))
    assert tracker.static(static(), 4.2, (4.1, 4.2))
    assert tracker.current is None
    assert tracker.records[0].close_reason == "confirmed_after_score"


def test_confirmed_pair_waits_for_s3_grace(tracker: object) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 3, 700)
    tracker.finish_frame(3.1)
    assert tracker.source == "S1"
    assert not tracker.static(static(), 3.1, (3.1, 3.1))
    tracker.finish_frame(3 + S3_END_QUIET_SEC)
    assert tracker.source == "S3"
    assert not tracker.static(static(), 3 + S3_END_QUIET_SEC, (4, 4))
    assert tracker.static(static(), 4, (4, 4))


def test_confirmed_pair_does_not_replace_missing_end_signal(tracker: object) -> None:
    fire(tracker)
    tracker.finalize("1P", 3, 700)
    assert not tracker.static(static(), 3.1, (3.1, 3.1))
    assert tracker.current is not None


@pytest.mark.parametrize("per_side", [False, True])
def test_static_every_settled_frame_uses_confirmed_pair(per_side: bool) -> None:
    calls = []
    def build(boards: tuple, snapshot: object, elapsed: float, m0: float) -> object:
        calls.append((elapsed, boards[1]._grid.copy()))
        return build_static(boards, snapshot, elapsed, m0)
    overlay = ExchangeEventOverlay(Models(), build, Signals, lambda b, q: .5, per_side)
    counts = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                             chain_total_score_p1=0, chain_total_score_p2=0)
    for frame in range(3):
        overlay.update(result(frame / 30), object(), counts, frame / 30, 1)
    observed = result(.1)
    observed.p2.state = BoardState.CHAIN
    observed.p2.confirmed_board._grid[:] = 9
    overlay.update(observed, object(), counts, .1, 1)
    assert len(calls) == (4 if per_side else 3)
    assert not calls[-1][1].any()


def test_formula_visibility_is_current_read_only() -> None:
    pipeline = SimpleNamespace(_formula_last_read_1p=SimpleNamespace(valid=True),
                               _formula_last_read_2p=SimpleNamespace(valid=False))
    assert formula_visible_from_pipeline(pipeline) == (True, False)
    pipeline._formula_last_read_1p = None
    assert formula_visible_from_pipeline(pipeline) == (False, False)
