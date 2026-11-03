"""E10cの物理余裕・得点帰属・保持中の確定盤面再判定を検証する。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import exchange_event_landing as landing
from src.board import Board
from src.exchange_event_overlay import ConfirmedSide
from tests.test_e10_exchange_landing import board_at_height
from tests.test_e10b_exchange_landing import setup_projection
from tests.test_exchange_event_tracker import tracker


@pytest.mark.parametrize("incoming,death", [(6, False), (11, False), (12, True), (17, True)])
def test_full_row_margin_ignores_random_remainder(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                                incoming: int, death: bool) -> None:
    projection, overlay, _, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    value = projection._evaluate(overlay, snapshot, tuple(h[-1] for h in overlay._history),
                                 [0, incoming], (1, 1), dict(p1=.8), 2)
    assert (value["source"] == "unavoidable_death") is death


@pytest.mark.parametrize("reason,formula,death", [
    ("display_stable", None, False), ("score_finalize", None, True),
    ("formula_match", 1400, True), ("display_stable", 1400, True),
])
def test_attack_needs_attributed_score(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                     reason: str, formula: int | None, death: bool) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    chain = tracker.latest_chain("1P")
    chain.formula_total, chain.score_ready_reason = formula, reason
    chain.score_ready_sec, chain.score_delta = 2, 1400
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    assert (projection.death is not None) is death


@pytest.mark.parametrize("incoming_zero", [False, True])
def test_receiver_clear_releases_without_new_chain(tracker: object, monkeypatch: pytest.MonkeyPatch,
                                                  incoming_zero: bool) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    assert projection.death is not None
    overlay._history[1].append(ConfirmedSide(2.1, Board(), np.ones(4, dtype=int)))
    if incoming_zero:
        snapshot.total_dropped_to_p2 = 20
    projection.update(overlay, result, snapshot, 2.1)
    assert projection.death is None
    assert tracker.probability != .98


def test_same_board_with_new_timestamp_keeps_latch(tracker: object,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    overlay._history[1].append(ConfirmedSide(2.1, board_at_height(11), np.ones(4, dtype=int)))
    projection.update(overlay, result, snapshot, 2.1)
    assert not projection.board_changed and projection.death is not None


@pytest.mark.parametrize("score,accepted", [(40, False), (120, True)])
def test_color_disappearance_respects_score_lower_bound(tracker: object,
        monkeypatch: pytest.MonkeyPatch, score: int, accepted: bool) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    board = board_at_height(11)
    board._grid[-6:, 0] = (1, 2, 1, 2, 1, 2)
    overlay._history[1] = [ConfirmedSide(1, board, np.ones(4, dtype=int))]
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    chain = SimpleNamespace(formula_total=score, score_delta=score)
    monkeypatch.setattr(tracker, "latest_chain", lambda side: chain)
    overlay._history[1].append(ConfirmedSide(2.1, Board(), np.ones(4, dtype=int)))
    projection._select_boards(overlay, 2.1)
    assert projection.board_changed is accepted
    assert bool(projection.rejected_boards) is (not accepted)


def test_one_hand_does_not_borrow_free_third_move() -> None:
    """実A盤面: 1手では発火不能、3手なら大連鎖という時間予算の差を固定する。"""
    board = Board()
    board._grid = np.array([
        [1, 0, 0, 0, 0, 0], [1, 3, 0, 0, 0, 4], [5, 3, 5, 9, 5, 4],
        [4, 5, 9, 9, 5, 5], [4, 3, 5, 4, 4, 1], [1, 5, 5, 4, 3, 1],
        [3, 3, 3, 5, 5, 1], [1, 1, 4, 5, 1, 4], [1, 4, 4, 3, 3, 4],
        [5, 5, 5, 3, 5, 3], [1, 4, 3, 5, 5, 4], [1, 1, 4, 3, 3, 5],
        [4, 4, 3, 5, 4, 4]], dtype=np.int8)
    raw = board._grid
    args = (raw.tobytes(), raw.shape, raw.dtype.str, (1, 1, 5, 1))
    assert landing.future_send(*args, 1, 0) == 0
    assert landing.future_send(*args, 3, 0) > 0


def test_own_chain_time_cannot_be_used_for_placements(monkeypatch: pytest.MonkeyPatch) -> None:
    """相手の残り3秒のうち自側消去2秒は操作できず、残り1秒と最後の1手だけ。"""
    monkeypatch.setattr(landing, "estimate_chain_anim_duration_sec", lambda *args: 3)
    assert landing.remaining_hands(2, 0, 0, busy_sec=2) == 2
    assert landing.remaining_hands(2, 0, 0, busy_sec=4) == 1


def test_cancellation_zero_releases_without_board_change(tracker: object,
        monkeypatch: pytest.MonkeyPatch) -> None:
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    tracker.latest_chain("1P").formula_total = 0
    projection.update(overlay, result, snapshot, 2.1)
    assert projection.death is None and tracker.probability != .98


def test_short_search_cannot_prove_new_unavoidable_death(tracker: object,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """未検知設置・枝刈りでn手探索から落ちた応手が既存候補にあれば初回断定しない。"""
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0 if args[4] == 1 else 100)
    projection.update(overlay, result, snapshot, 2)
    assert projection.last["hands"] == (1, 1)
    assert projection.last["optimistic_send"][1] == 100
    assert projection.death is None


def test_real_placement_rechecks_actual_budget_after_latch(tracker: object,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """保持中は実手数内で生存できるかを再計算し、遠い将来の手を着弾前に借りない。"""
    projection, overlay, result, snapshot = setup_projection(tracker)
    monkeypatch.setattr(landing, "future_send", lambda *args: 0)
    projection.update(overlay, result, snapshot, 2)
    changed = board_at_height(11)
    changed._grid[-1, 0] = 9
    overlay._history[1].append(ConfirmedSide(2.1, changed, np.ones(4, dtype=int)))
    monkeypatch.setattr(landing, "future_send", lambda *args: 0 if args[4] == 1 else 100)
    projection.update(overlay, result, snapshot, 2.1)
    assert projection.board_changed
    assert projection.death is not None and tracker.probability == .98
