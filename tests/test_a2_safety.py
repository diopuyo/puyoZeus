"""A2の段原点とMENU開始受理の過大方向を検査する。"""
from types import SimpleNamespace as NS
import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_formula_origin import FormulaOrigin
from src.exchange_history_integrity import MatchHistoryIntegrity, BOUNDARY_ZERO_MAX_SEC


def side(state: str = 'MENU', score: int | None = 0, cells: int = 0,
         count: int | None = None, total: int = 100, trigger: float = 1.) -> NS:
    """必要な観測だけを用意する。"""
    board = Board()
    board._grid.flat[:cells] = 1
    event = None if count is None else NS(mechanism='formula_read', chain_count=count,
                                        total_score=total, trigger_sec=trigger)
    return NS(state=BoardState[state], score=score, confirmed_board=board, chain_event=event)


def history(enabled: bool = True) -> MatchHistoryIntegrity:
    """前試合の両側の得点を観測しておく。"""
    obj = MatchHistoryIntegrity(boundary_score_start=True, strict_integrity=True,
                                menu_boundary_start=enabled, retain_audit=True)
    obj.observe((side('STABLE', 100, 10), side('STABLE', 200, 10)), 0, 0.)
    return obj


@pytest.mark.parametrize('enabled', [False, True])
def test_menu_start_flag(enabled: bool) -> None:
    obj = history(enabled)
    obj.observe((side(), side()), 1, 1.)
    assert obj.evidence()['history_complete'] is enabled


@pytest.mark.parametrize('state,cells,accepted', [('STABLE', 0, True), ('STABLE', 2, True),
    ('STABLE', 3, False), ('CHAIN', 0, False), ('GRAVITY_SETTLE', 0, False)])
def test_mixed_side(state: str, cells: int, accepted: bool) -> None:
    obj = history()
    obj.observe((side(), side(state, 0, cells)), 1, 1.)
    assert obj.evidence()['history_complete'] is accepted


@pytest.mark.parametrize('state', ['STABLE', 'CHAIN', 'OJAMA_FALL'])
def test_false_boundary_then_menu_zero(state: str) -> None:
    obj = history()
    obj.observe((side(state, None, 10), side()), 1, 1.)
    obj.observe((side(), side()), 1, 1.1)
    assert not obj.evidence()['history_complete']


@pytest.mark.parametrize('delay,accepted', [(BOUNDARY_ZERO_MAX_SEC, True),
                                         (BOUNDARY_ZERO_MAX_SEC + .01, False)])
def test_zero_deadline(delay: float, accepted: bool) -> None:
    obj = history()
    obj.observe((side(score=None), side(score=None)), 1, 1.)
    obj.observe((side(), side()), 1, 1.+delay)
    assert obj.evidence()['history_complete'] is accepted


@pytest.mark.parametrize('game', [0, 2])
def test_false_boundary_number(game: int) -> None:
    obj = history()
    obj.observe((side(), side()), game, 1.)
    assert not obj.evidence()['history_complete']


def test_zero_misread_after_positive() -> None:
    obj = history()
    obj.observe((side(score=5), side()), 1, 1.)
    obj.observe((side(), side()), 1, 1.1)
    assert not obj.evidence()['history_complete']


@pytest.mark.parametrize('previous', [(None, 100), (0, 100), (100, 0)])
def test_previous_score_evidence(previous: tuple) -> None:
    obj = MatchHistoryIntegrity(boundary_score_start=True, strict_integrity=True,
                                menu_boundary_start=True)
    obj.observe(tuple(side('MENU', s) for s in previous), 0, 0.)
    obj.observe((side(), side()), 1, 1.)
    assert not obj.evidence()['history_complete']


@pytest.mark.parametrize('placement', [0, 2, 40, -1])
def test_formula_origin_uses_placement(placement: int) -> None:
    obj = FormulaOrigin()
    obj.observe(side('CHAIN', count=1), 100, None, True, 1.)
    obj.observe(side('STABLE'), 100, 489, False, 2.)
    obj.observe(side('STABLE'), 100, 489+placement, False, 2.1)
    raw = side('CHAIN', count=2, total=200, trigger=3.)
    corrected, formula = obj.observe(raw, 200, None, True, 3.)
    adjusted = placement == 2
    assert corrected.chain_event.chain_count == (1 if adjusted else 2)
    assert formula == (100 if adjusted else 200)
    assert raw.chain_event.chain_count == 2
    later, formula = obj.observe(side('CHAIN', count=3, total=520, trigger=4.),
                                520, None, True, 4.)
    assert later.chain_event.chain_count == (2 if adjusted else 3)
    assert formula == (420 if adjusted else 520)


def test_no_stable_no_rebase() -> None:
    obj = FormulaOrigin()
    obj.observe(side('CHAIN', count=1), 100, None, True, 1.)
    obj.observe(side('GRAVITY_SETTLE'), 100, 489, False, 2.)
    obj.observe(side('GRAVITY_SETTLE'), 100, 491, False, 2.1)
    value, _ = obj.observe(side('CHAIN', count=2, total=200, trigger=3.), 200, None, True, 3.)
    assert value.chain_event.chain_count == 2


@pytest.mark.parametrize('count,total', [(1, 100), (3, 520), (2, 50), (2, 100)])
def test_missing_or_reset_formula_never_guesses(count: int, total: int) -> None:
    obj = FormulaOrigin()
    obj.observe(side('CHAIN', count=1), 100, None, True, 1.)
    obj.observe(side('STABLE'), 100, 489, False, 2.)
    obj.observe(side('STABLE'), 100, 491, False, 2.1)
    value, _ = obj.observe(side('CHAIN', count=count, total=total, trigger=3.),
                           total, None, True, 3.)
    assert value.chain_event.chain_count == count
    assert not obj.audit


def test_frozen_live_side_and_event() -> None:
    from dataclasses import make_dataclass
    obj = FormulaOrigin()
    raw = side('CHAIN', count=1)
    event_type = make_dataclass('Event', [(k, type(v)) for k, v in vars(raw.chain_event).items()], frozen=True)
    raw.chain_event = event_type(**vars(raw.chain_event))
    side_type = make_dataclass('Side', [(k, type(v)) for k, v in vars(raw).items()], frozen=True)
    value, _ = obj.observe(side_type(**vars(raw)), 100, None, True, 1.)
    assert value.chain_event.chain_count == 1


def test_formula_session_resets_after_rebase() -> None:
    obj = FormulaOrigin()
    obj.observe(side('CHAIN', count=1), 100, None, True, 1.)
    obj.observe(side('STABLE'), 100, 489, False, 2.)
    obj.observe(side('STABLE'), 100, 491, False, 2.1)
    obj.observe(side('CHAIN', count=2, total=200, trigger=3.), 200, None, True, 3.)
    value, formula = obj.observe(side('CHAIN', count=1, total=40, trigger=8.), 40, None, True, 8.)
    assert (value.chain_event.chain_count, formula) == (1, 40)


def test_menu_zero_without_previous_game_never_accepts() -> None:
    obj = MatchHistoryIntegrity(boundary_score_start=True, strict_integrity=True,
                                menu_boundary_start=True)
    obj.observe((side(), side()), 0, 0.)
    assert not obj.evidence()['history_complete']


def test_raw_formula_session_spans_short_placement_gap() -> None:
    from src.score_ocr import FormulaStepAccumulator
    accum = FormulaStepAccumulator()
    formula = NS(valid=True, left=50, right=2)
    absent = NS(valid=False, left=None, right=None)
    for stamp in (4259.633333333, 4259.666666667, 4260.5):
        accum.update(stamp, formula)
    for stamp in (4260.533333333, 4260.566666667, 4262.1):
        accum.update(stamp, absent, score_displayed=True)
    for stamp in (4262.366666667, 4262.4):
        accum.update(stamp, formula)
    assert (accum.step_count, accum.total_power) == (2, 200)


@pytest.mark.parametrize('visible,board', [(True, True), (False, False)])
def test_partial_stable_evidence_does_not_rebase(visible: bool, board: bool) -> None:
    obj = FormulaOrigin()
    obj.observe(side('CHAIN', count=1), 100, None, True, 1.)
    stable = side('STABLE')
    if not board:
        stable.confirmed_board = None
    obj.observe(stable, 100, 489, visible, 2.)
    obj.observe(stable, 100, 491, visible, 2.1)
    value, _ = obj.observe(side('CHAIN', count=2, total=200, trigger=3.), 200, None, True, 3.)
    assert value.chain_event.chain_count == 2


def test_same_first_score_after_session_reset_consumes_boundary() -> None:
    obj = FormulaOrigin()
    obj.observe(side('CHAIN', count=1), 100, None, True, 1.)
    obj.observe(side('STABLE'), 100, 489, False, 2.)
    obj.observe(side('STABLE'), 100, 491, False, 2.1)
    obj.observe(side('CHAIN', count=1, total=100, trigger=4.), 100, None, True, 4.)
    value, total = obj.observe(side('CHAIN', count=2, total=420, trigger=4.), 420, None, True, 5.)
    assert (value.chain_event.chain_count, total) == (2, 420)
    assert not obj.audit
