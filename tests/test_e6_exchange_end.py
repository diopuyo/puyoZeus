"""通常落下加点と連鎖再開を分離するE6回帰検証。"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from scripts.visualize_advantage_overlay import _ExchangeEventEndSignals
from src.board_state_machine import BoardState
from src.exchange_event_overlay import ExchangeEventOverlay

from src.exchange_event_tracker import ExchangeEventTracker, S3_END_QUIET_SEC
from src.ojama_accounting import CHAIN_TOTAL_MIN_SCORE
from tests.test_exchange_event_tracker import fire, tracker
from tests.test_exchange_event_tracker import Models
from tests.test_exchange_event_overlay import Signals, build_static, result


@pytest.mark.parametrize("delta", [1, 2, CHAIN_TOTAL_MIN_SCORE - 1])
def test_drop_points_do_not_revoke_end(tracker: ExchangeEventTracker, delta: int) -> None:
    """終了後の操作加点はS3の根拠を取り消さない。"""
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100, 700)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 3, 700)
    tracker.finish_frame(3 + S3_END_QUIET_SEC)
    tracker.observe_score("1P", 4, 800 + delta, 100, 700)
    assert tracker.source == "S3"  # 落下加点を基準へ吸収し、連鎖得点の確定を保持する。
    assert tracker.latest_chain("1P").end_signal_sec == 3
    assert "revoked_sec" not in tracker.latest_chain("1P").end_signals[0]


@pytest.mark.parametrize("delta", [CHAIN_TOTAL_MIN_SCORE, 700])
def test_erasure_points_still_revoke_end(tracker: ExchangeEventTracker, delta: int) -> None:
    """最小消去以上の加点は従来どおり再開を検出する。"""
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100, 700)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 4, 800 + delta, 100, 700)
    assert tracker.latest_chain("1P").end_signal_sec is None
    assert tracker.latest_chain("1P").end_signals[0]["revoked_sec"] == 4


def test_small_steps_do_not_accumulate_into_false_chain(tracker: ExchangeEventTracker) -> None:
    """複数ツモの落下加点を合算して消去得点にしない。"""
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100, 700)
    tracker.end("1P", 3, "next")
    for step in range(CHAIN_TOTAL_MIN_SCORE + 1):
        tracker.observe_score("1P", 3 + step / 30, 800 + step, 100, 700)
    assert tracker.latest_chain("1P").end_signal_sec == 3


def sensor() -> tuple:
    """開始設置の移動後に基準化した、実物の終了センサーを作る。"""
    observed = result(1)
    snapshot = SimpleNamespace(total_dropped_to_p1=0, total_dropped_to_p2=0)
    signals = _ExchangeEventEndSignals(0, observed, snapshot, 1)
    signals.update(observed, snapshot, 2)
    signals.update(observed, snapshot, 2 + 1 / 30)
    return signals, observed, snapshot


@pytest.mark.parametrize("pair", [(9, 2), (0, 2), (10, 2), None])
def test_invalid_next_cannot_end_chain(pair: tuple | None) -> None:
    """おじゃま・欠測等をNEXT色変化にしない。"""
    signals, observed, snapshot = sensor()
    observed.p1.state = BoardState.GRAVITY_SETTLE
    observed.p1.next_pair = pair
    for frame in range(10):
        assert signals.update(observed, snapshot, 4 + frame / 30) is None


@pytest.mark.parametrize("reason", ["next", "slide", "ojama"])
def test_own_physical_signal_ends_chain(reason: str) -> None:
    signals, observed, snapshot = sensor()
    observed.p1.state = BoardState.STABLE
    if reason == "next":
        observed.p1.next_pair = (3, 4)
    elif reason == "slide":
        observed.p1.next_slide_motion = True
    else:
        observed.p1.state = BoardState.OJAMA_FALL
        snapshot.total_dropped_to_p1 = 30
    assert signals.update(observed, snapshot, 4) is None
    assert signals.update(observed, snapshot, 4 + 1 / 30) == reason


@pytest.mark.parametrize("kind", ["opponent_next", "opponent_fall", "account", "tsumo"])
def test_nonphysical_or_opponent_signal_cannot_end_chain(kind: str) -> None:
    signals, observed, snapshot = sensor()
    observed.p1.state = BoardState.STABLE
    if kind == "opponent_next":
        observed.p2.next_pair = (3, 4)
        observed.p2.next_slide_motion = True
    elif kind == "opponent_fall":
        observed.p2.state = BoardState.OJAMA_FALL
        snapshot.total_dropped_to_p2 = 30
    elif kind == "account":
        snapshot.total_dropped_to_p1 = 30
    elif kind == "tsumo":
        observed.p1.state = BoardState.TSUMO_FALL
    for frame in range(10):
        assert signals.update(observed, snapshot, 4 + frame / 30) is None


def test_own_fall_does_not_wait_for_landing_account() -> None:
    """落下開始と着地会計の増加は別時刻。絶対律の終了は前者で観測する。"""
    signals, observed, snapshot = sensor()
    observed.p1.state = BoardState.OJAMA_FALL
    assert snapshot.total_dropped_to_p1 == 0
    assert signals.update(observed, snapshot, 4) is None
    assert signals.update(observed, snapshot, 4 + 1 / 30) == "ojama"


@pytest.mark.parametrize("visible,ended", [((True, False), False), ((False, True), True)])
def test_current_formula_blocks_only_own_end(visible: tuple, ended: bool) -> None:
    """同一フレームの式表示を終了通知より先に評価する。"""
    overlay = ExchangeEventOverlay(Models(), build_static, Signals)
    counts = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                             chain_total_score_p1=0, chain_total_score_p2=0)
    for t in (0, 1):
        overlay.update(result(t), object(), counts, t, 1)
    overlay.update(result(2), object(), counts, 2, 1, formula_visible=visible)
    assert (overlay.tracker.latest_chain("1P").end_signal_sec is not None) == ended


def test_drop_point_preserves_score_stability(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, 800, 100, 700)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 3, 700)
    tracker.observe_score("1P", 3.1, 801, 100, 700)
    chain = tracker.latest_chain("1P")
    assert chain.end_signal_sec == 3 and chain.score_ready_sec == 3
    assert chain.score_delta == 700 and chain.score_before == 101


def test_accounting_followup_does_not_mask_valid_next() -> None:
    """会計追随を除外しても、同時に観測した本物のNEXT移動は採用する。"""
    signals, observed, snapshot = sensor()
    observed.p1.state = BoardState.STABLE
    observed.p1.next_pair = (3, 4)
    snapshot.total_dropped_to_p1 = 30
    assert signals.update(observed, snapshot, 4) is None
    assert signals.update(observed, snapshot, 4 + 1 / 30) == "next"


def test_next_colour_change_requires_queue_successor() -> None:
    """演出による通常色同士の誤読を、NEXTの物理送りと混同しない。"""
    signals, observed, snapshot = sensor()
    observed.p1.state = BoardState.GRAVITY_SETTLE
    observed.p1.next_pair = (1, 5)
    for frame in range(10):
        assert signals.update(observed, snapshot, 4 + frame / 30) is None
    observed.p1.next_pair = observed.p1.dnext_pair
    assert signals.update(observed, snapshot, 5) is None
    assert signals.update(observed, snapshot, 5 + 1 / 30) == "next"


def test_slide_waits_through_formula_session_gap() -> None:
    """式の幕間は認識器と同じ物理時間を使い、slide単独の終了を抑止する。"""
    from src.score_ocr import FORMULA_SESSION_RESET_SEC

    class Slide(Signals):
        def update(self, r: object, snapshot: object, t_sec: float) -> str | None:
            return "slide" if t_sec >= 2 else None

    overlay = ExchangeEventOverlay(Models(), build_static, Slide)
    counts = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                             chain_total_score_p1=0, chain_total_score_p2=0)
    for t in (0, 1):
        overlay.update(result(t), object(), counts, t, 1)
    last_formula = 1.5
    overlay.update(result(last_formula), object(), counts, last_formula, 1,
                   formula_visible=(True, False))
    end = last_formula + FORMULA_SESSION_RESET_SEC
    for t in (2, end - 1 / 30):
        overlay.update(result(t), object(), counts, t, 1)
        assert overlay.tracker.latest_chain("1P").end_signal_sec is None
    overlay.update(result(end), object(), counts, end, 1)
    assert overlay.tracker.latest_chain("1P").end_signal_sec == end
