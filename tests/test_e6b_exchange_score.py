"""落下加点でS3確定・10フレーム不変・安全弁をリセットしない検証。"""
from __future__ import annotations

import pytest
import numpy as np

from src.chain_id_resolver import ChainObservation, ObservationKind
from src.exchange_event_features import SIDE_COLUMNS

from src.exchange_event_tracker import (
    ExchangeEventTracker, EXCHANGE_IDLE_TIMEOUT_SEC, OBSERVATION_FPS,
    S3_SCORE_STABLE_FRAMES, S3_END_QUIET_SEC,
)
from tests.test_exchange_event_tracker import fire, static, tracker


@pytest.mark.parametrize("step", [1, 2])
def test_drop_points_count_as_unchanged_chain_score(tracker: ExchangeEventTracker, step: int) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100)
    tracker.end("1P", 3, "next")
    for frame in range(1, S3_SCORE_STABLE_FRAMES + 1):
        tracker.observe_score("1P", 3 + frame / OBSERVATION_FPS, 800 + step * frame, 100)
    chain = tracker.latest_chain("1P")
    assert chain.score_delta == 700
    assert chain.stable_frames == S3_SCORE_STABLE_FRAMES
    assert chain.score_before == 100 + step * S3_SCORE_STABLE_FRAMES
    tracker.finish_frame(3 + S3_END_QUIET_SEC)
    assert tracker.source == "S3"


def test_drop_points_do_not_extend_idle_timeout(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100, 700)
    tracker.end("1P", 3, "next")
    for frame in range(int(EXCHANGE_IDLE_TIMEOUT_SEC * OBSERVATION_FPS)):
        t_sec = 3 + frame / OBSERVATION_FPS
        tracker.observe_score("1P", t_sec, 800 + frame, 100, 700)
        tracker.finish_frame(t_sec)
    assert tracker.source == "S3" and tracker._last_activity_sec == 3
    tracker.finish_frame(3 + EXCHANGE_IDLE_TIMEOUT_SEC)
    assert tracker.current is None
    assert tracker.records[0].close_reason == "activity_timeout"


def test_pre_end_drop_is_not_chain_activity(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 100, 100)
    tracker.observe_score("1P", 4, 101, 100)
    assert tracker._last_activity_sec == 2
    assert tracker.latest_chain("1P").score_before == 100


def test_drop_baseline_survives_real_resumption(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3.1, 801, 100)
    tracker.observe_score("1P", 4, 841, 100)
    chain = tracker.latest_chain("1P")
    assert chain.end_signal_sec is None
    assert chain.score_before == 101 and chain.drop_bonus_score == 1
    assert tracker._last_activity_sec == 4


def test_resolver_fragment_does_not_leave_unobserved_participant(tracker: ExchangeEventTracker) -> None:
    """物理終了前の段数訂正による新IDを、同じ参加連鎖として観測し続ける。"""
    for t_sec, count, total in ((2, 3, 1000), (3, 1, 40)):
        tracker.fire(t_sec=t_sec, triggers=(t_sec, None), static=static(),
            prefire_sides=np.zeros((2, len(SIDE_COLUMNS))), score_elapsed_sec=t_sec,
            observations=(ChainObservation("1P", t_sec, ObservationKind.FORMULA_STEP,
                                           chain_count=count, total_score=total),))
    assert len(tracker.current.chains) == 1
    assert tracker.resolver.active()[0].chain_id != tracker.current.chains[0].chain_id
    tracker.end("1P", 4, "next")
    tracker.finalize("1P", 5, 1000)
    assert not tracker.resolver.active()
    tracker.finish_frame(5)
    assert tracker.source == "S3"
    assert tracker.current.chains[0].score_delta == 1000


def test_real_end_allows_another_participant(tracker: ExchangeEventTracker) -> None:
    """同側でも物理終了後の新しい連鎖を一つへ潰さない。"""
    for t_sec in (2, 4):
        tracker.fire(t_sec=t_sec, triggers=(t_sec, None), static=static(),
            prefire_sides=np.zeros((2, len(SIDE_COLUMNS))), score_elapsed_sec=t_sec,
            observations=(ChainObservation("1P", t_sec, ObservationKind.FORMULA_STEP,
                                           chain_count=1, total_score=40),))
        tracker.end("1P", t_sec + .1, "next")
        tracker.finalize("1P", t_sec + .2, 40)
    assert len(tracker.current.chains) == 2


def test_placement_separates_new_formula_from_closed_exchange(tracker: ExchangeEventTracker) -> None:
    """NEXT終了後に実際の落下加点を挟んだ単発は、式の段数が増えても別連鎖。"""
    kwargs = dict(static=static(), prefire_sides=np.zeros((2, len(SIDE_COLUMNS))),
                  score_elapsed_sec=2.0)
    tracker.fire(t_sec=2, triggers=(2, None), observations=(ChainObservation(
        "1P", 2, ObservationKind.FORMULA_STEP, 1, 40),), **kwargs)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3, 140, 100, 40)
    tracker.finish_frame(3 + S3_END_QUIET_SEC)
    tracker.static(static(), 3.4, (3.4, 3.4))
    tracker.note_placement("1P", 3.5)
    tracker.fire(t_sec=4, triggers=(4, None), observations=(ChainObservation(
        "1P", 4, ObservationKind.FORMULA_STEP, 2, 80),), **kwargs)
    assert len(tracker.records) == 2
    assert tracker.records[0].chains[0].end_signal_sec == 3
    assert "revoked_sec" not in tracker.records[0].chains[0].end_signals[0]
    assert tracker.current.chains[0].chain_id != tracker.records[0].chains[0].chain_id


def test_placement_does_not_extend_timeout_via_next_formula(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.note_placement("1P", 3.5)
    tracker.activity("1P", 4)
    assert tracker._last_activity_sec == 3
    assert tracker.latest_chain("1P").end_signal_sec == 3
