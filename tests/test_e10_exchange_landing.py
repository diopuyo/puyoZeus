"""E10の物理換算・仮想着弾・logit合成・回避可能時の非固定を検証する。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.board import Board, DEATH_COL, DEATH_ROW, COLOR_OJAMA
from src import exchange_event_landing as landing
from src.exchange_event_overlay import ConfirmedSide
from tests.test_exchange_event_tracker import fire, static, tracker


def board_at_height(height: int) -> Board:
    """色連鎖のない確定おじゃま列を作る。"""
    board = Board()
    board._grid[-height:, DEATH_COL] = COLOR_OJAMA
    return board


@pytest.mark.parametrize("probability", [.02, .2, .5, .8, .98])
def test_logit_mean_symmetric(probability: float) -> None:
    assert landing.logit_mean(probability, 1-probability) == pytest.approx(.5)
    assert landing.logit_mean(probability, probability) == pytest.approx(probability)


def test_logit_mean_differs_from_probability_mean() -> None:
    assert landing.logit_mean(.9, .5) == pytest.approx(.75)


@pytest.mark.parametrize("remaining, expected", [(0, 1), (.732, 1), (.733, 2), (1.466, 3)])
def test_time_budget_has_one_landing_hand(monkeypatch: pytest.MonkeyPatch,
                                        remaining: float, expected: int) -> None:
    monkeypatch.setattr(landing, "estimate_chain_anim_duration_sec", lambda *args: remaining)
    assert landing.remaining_hands(2, 0, 0) == expected


@pytest.mark.parametrize("height,incoming", [(11, 6), (10, 12), (9, 18), (8, 36)])
def test_minimum_cancel_reuses_physical_landing(height: int, incoming: int) -> None:
    board, opponent = board_at_height(height), Board()
    before = board._grid.copy()
    required = landing.minimum_cancel(board, opponent, incoming)
    assert required > 0
    safe, _, _ = landing.land_pending_ojama_onto_board(board, opponent, incoming-required)
    assert not safe.is_dead()
    np.testing.assert_array_equal(board._grid, before)


def test_hidden_row_is_not_death() -> None:
    board = Board()
    board._grid[DEATH_ROW-1, DEATH_COL] = COLOR_OJAMA
    assert not board.is_dead()


def test_score_net_and_dropped_are_not_added_twice(tracker: object) -> None:
    fire(tracker, triggers=(2, 2))
    tracker.observe_score("1P", 2, None, formula_total=1270)
    tracker.observe_score("2P", 2, None, formula_total=540)
    projection = landing.ExchangeLandingProjection()
    projection.drops = (14, 12)
    assert projection._incoming(tracker, (14, 12)) == [0, 11]
    assert projection._incoming(tracker, (14, 18)) == [0, 5]
    assert projection._incoming(tracker, (14, 24)) == [0, 0]


@pytest.mark.parametrize("available,fixed", [(0, True), (100, False)])
def test_death_requires_counter_shortfall(monkeypatch: pytest.MonkeyPatch,
                                          tracker: object, available: int, fixed: bool) -> None:
    fire(tracker)
    tracker.observe_score("1P", 2, None, formula_total=1400)
    monkeypatch.setattr(landing, "future_send", lambda *args: available)
    overlay = SimpleNamespace(tracker=tracker, _start=0, _m0=lambda *args: .5,
                              _build_static=lambda *args: static())
    boards = (Board(), board_at_height(11))
    latest = tuple(ConfirmedSide(1, b, np.ones(4, dtype=int)) for b in boards)
    projection = landing.ExchangeLandingProjection()
    result = projection._evaluate(overlay, object(), latest, [0, 20], (1, 1), {"p1": .8}, 2)
    assert (result["source"] == "unavoidable_death") == fixed
    assert result["p1"] == pytest.approx(.98 if fixed else landing.logit_mean(.8, .4))


def test_absent_attack_does_not_fix_dead_board(monkeypatch: pytest.MonkeyPatch,
                                             tracker: object) -> None:
    fire(tracker)
    overlay = SimpleNamespace(tracker=tracker, _start=0, _m0=lambda *args: .5,
                              _build_static=lambda *args: static())
    latest = tuple(ConfirmedSide(1, board_at_height(12), np.ones(4, dtype=int)) for _ in range(2))
    result = landing.ExchangeLandingProjection()._evaluate(
        overlay, object(), latest, [0, 0], (1, 1), {"p1": .8}, 2)
    assert result["dead_sides"] == []


def test_unchanged_quantity_rechecks_confirmed_board_change(
        monkeypatch: pytest.MonkeyPatch, tracker: object) -> None:
    """E10c: 受け量が同じでもSTABLE確定盤面の変化を再判定する。"""
    fire(tracker)
    tracker.observe_score("1P", 2, None, formula_total=700)
    tracker.finish_frame(2)
    snapshot = SimpleNamespace(total_dropped_to_p1=0, total_dropped_to_p2=0)
    histories = [[ConfirmedSide(1, Board(), np.ones(4, dtype=int))] for _ in range(2)]
    overlay = SimpleNamespace(tracker=tracker, _m0=object(), _history=histories,
                              _snapshots=[(1, snapshot)])
    projection, calls = landing.ExchangeLandingProjection(), []
    def evaluate(*args: object) -> dict:
        calls.append(args)
        return dict(source="unavoidable_death", p1=.98, t_sec=args[-1], dead_sides=["2P"])
    monkeypatch.setattr(projection, "_evaluate", evaluate)
    result = SimpleNamespace(p1=SimpleNamespace(chain_event=None), p2=SimpleNamespace(chain_event=None))
    projection.update(overlay, result, snapshot, 2)
    histories[1].append(ConfirmedSide(2.1, board_at_height(3), np.ones(4, dtype=int)))
    projection.update(overlay, result, snapshot, 2.1)
    assert len(calls) == 2 and tracker.probability == .98
    tracker.observe_score("1P", 2.2, None, formula_total=1400)
    tracker.finish_frame(2.2)
    projection.update(overlay, result, snapshot, 2.2)
    assert len(calls) == 3


def test_lit_group_at_death_row_is_resolved_before_death(tracker: object) -> None:
    """天井設置と同時の発火を、近未来探索のdead早期returnで捨てない。"""
    fire(tracker)
    tracker.observe_score("1P", 2, None, formula_total=100)
    board = Board()
    board._grid[DEATH_ROW:DEATH_ROW+8, DEATH_COL] = 1
    latest = (ConfirmedSide(1, board, np.ones(4, dtype=int)),
              ConfirmedSide(1, Board(), np.ones(4, dtype=int)))
    resolved, credit = landing.ExchangeLandingProjection()._receivers(tracker, latest, [6, 0])
    assert board.is_dead() and not resolved[0].is_dead()
    assert credit == [4, 0]  # 8個消去400点から観測済み100点を除いた300点。


def test_ceiling_ignition_is_a_valid_response_only_on_opt_in() -> None:
    """天井マスへ置いて即消しする応手を、設置直後の窒息で捨てない。"""
    board = Board()
    board._grid = np.array([
        [4, 0, 0, 0, 0, 0], [2, 9, 0, 0, 0, 3], [1, 1, 1, 2, 0, 9],
        [2, 5, 5, 3, 9, 3], [3, 5, 3, 3, 1, 3], [2, 2, 9, 2, 2, 2],
        [2, 3, 2, 5, 1, 1], [3, 2, 5, 3, 3, 1], [3, 3, 2, 5, 5, 3],
        [2, 1, 2, 1, 2, 2], [3, 3, 5, 1, 2, 1], [1, 3, 1, 5, 1, 1],
        [1, 5, 5, 2, 2, 2]], dtype=np.int8)
    kwargs = dict(next_pair=(2, 5), dnext_pair=(5, 1), k_levels=(1,))
    before = landing.near_future_fire_power(board, **kwargs)
    after = landing.near_future_fire_power(board, **kwargs, resolve_before_death=True)
    assert before.values[1].raw == 0
    assert after.values[1].raw >= landing.minimum_cancel(board, Board(), 65)
    assert landing.near_future_fire_power(board, **kwargs).values[1] == before.values[1]


def test_response_retains_all_first_hand_placements() -> None:
    """8候補の枝刈りで落ちた77個の応手を、1手22配置の幅で保持する。"""
    board = Board()
    board._grid = np.array([
        [0, 0, 0, 0, 0, 0], [9, 2, 0, 0, 0, 9], [9, 1, 0, 0, 0, 9],
        [9, 9, 0, 0, 0, 9], [2, 9, 0, 0, 0, 9], [1, 9, 2, 0, 0, 9],
        [1, 9, 2, 1, 4, 9], [3, 3, 9, 1, 9, 9], [1, 1, 9, 2, 9, 1],
        [1, 4, 9, 2, 9, 3], [2, 2, 4, 9, 3, 1], [4, 2, 4, 2, 2, 3],
        [4, 1, 1, 2, 3, 3]], dtype=np.int8)
    grid = board._grid
    # 旧K=4は既知NEXT2手を含めた実6手である。
    response = landing.future_send(grid.tobytes(), grid.shape, grid.dtype.str, (3, 1, 4, 1), 6, 0)
    assert response >= landing.minimum_cancel(board, Board(), 84)
