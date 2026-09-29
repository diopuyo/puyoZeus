"""窒息baselineだけの限定、時刻保持、式・得点の独立確認を検証する。"""
from types import SimpleNamespace as NS
import pytest
from src.board import Board
from src.exchange_event_death_formula import DeathFormulaGuard, FORMULA_HOLD_MAX_SEC


def setup(cell: int = 1) -> tuple:
    """発火前の盤面と表示得点を用意する。"""
    guard, board = DeathFormulaGuard(), Board()
    board._grid[1, 2] = cell
    side = NS(score=100)
    result = NS(p1=side, p2=side, confirmed_dead_sides=())
    guard.observe(result, 0., 1, (100, 100), (False, False))
    history = [NS(t_sec=0., board=board)]
    event = NS(trigger_sec=.1, mechanism='baseline', chain_count=4, total_score=6200, before_board=board)
    return guard, result, history, event


def observe(guard: DeathFormulaGuard, result: NS, stamp: float = .2,
            score: float | None = 100, formula: bool = False) -> None:
    """現在フレームの観測を進める。"""
    guard.observe(result, stamp, 1, (score, 100), (formula, False))


@pytest.mark.parametrize('cell', [1, 2, 3, 4, 5, 9])
def test_occupied_baseline_held(cell: int) -> None:
    guard, result, history, event = setup(cell)
    observe(guard, result)
    assert guard.notifications(0, event, history, None) == []
    assert len(guard.audit) == 1


@pytest.mark.parametrize('cell', [0, 10])
def test_empty_or_unknown_not_changed(cell: int) -> None:
    guard, result, history, event = setup(cell)
    observe(guard, result)
    assert guard.notifications(0, event, history, None) == [event]
    assert not guard.audit


@pytest.mark.parametrize('mechanism', ['formula', 'formula_read', 'score_jump', 'landing', None])
def test_other_mechanisms_are_identical(mechanism: str | None) -> None:
    guard, result, history, event = setup()
    event.mechanism = mechanism
    observe(guard, result)
    assert guard.notifications(0, event, history, None)[0] is event
    assert not guard.audit


def test_hidden_row_not_guarded() -> None:
    guard, result, history, event = setup(0)
    history[0].board._grid[0, 2] = 1
    observe(guard, result)
    assert guard.notifications(0, event, history, None) == [event]


@pytest.mark.parametrize('score,formula,reason', [(100, True, 'confirmed_formula'), (101, False, 'confirmed_score')])
def test_confirmation_keeps_original_trigger(score: int, formula: bool, reason: str) -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    observe(guard, result, .3, score, formula)
    ready = guard.notifications(0, event, history, None)
    assert len(ready) == 1 and ready[0].trigger_sec == .1
    assert guard.audit[0]['outcome'] == reason


def test_formula_already_read_not_held() -> None:
    guard, result, history, event = setup()
    observe(guard, result, formula=True)
    assert guard.notifications(0, event, history, None) == [event]
    assert not guard.audit


def test_score_already_increased_not_held() -> None:
    guard, result, history, event = setup()
    observe(guard, result, score=101)
    assert guard.notifications(0, event, history, None) == [event]


@pytest.mark.parametrize('score', [None, float('nan'), -1, 99, 100])
def test_missing_or_unchanged_score_not_confirmation(score: float | None) -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    observe(guard, result, .3, score)
    assert not guard.notifications(0, None, history, None)


def test_timeout_discards_not_accepts() -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    observe(guard, result, .2+FORMULA_HOLD_MAX_SEC+.01, formula=True)
    assert not guard.notifications(0, event, history, None)
    assert guard.audit[0]['outcome'] == 'discarded_timeout'


def test_dead_discards_before_confirmation() -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    result.confirmed_dead_sides = ('1P',)
    observe(guard, result, .3, formula=True)
    assert not guard.notifications(0, event, history, None)
    assert guard.audit[0]['outcome'] == 'discarded_death'


def test_boundary_never_resends() -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    guard.reset()
    assert not guard.notifications(0, None, history, None)
    assert guard.audit[0]['outcome'] == 'discarded_boundary'


def test_repeated_notification_counted_once() -> None:
    guard, result, history, event = setup()
    for stamp in (.2, .3, .4):
        observe(guard, result, stamp)
        assert not guard.notifications(0, event, history, None)
    assert len(guard.audit) == 1


def test_future_board_not_used() -> None:
    guard, result, history, event = setup()
    history[0].t_sec = .15
    observe(guard, result)
    assert guard.notifications(0, event, history, None) == [event]


def test_existing_chain_continuation_unchanged() -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    assert guard.notifications(0, event, history, (.1, 'baseline', 4, 6200)) == [event]


def test_formula_notification_itself_is_unchanged_during_release() -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    formula = NS(trigger_sec=.3, mechanism='formula_read')
    observe(guard, result, .3, formula=True)
    ready = guard.notifications(0, formula, history, None)
    assert ready[0].trigger_sec == .1 and ready[1] is formula


def test_other_side_formula_cannot_release() -> None:
    guard, result, history, event = setup()
    observe(guard, result)
    guard.notifications(0, event, history, None)
    guard.observe(result, .3, 1, (100, 100), (False, True))
    assert not guard.notifications(0, None, history, None)


def test_read_between_trigger_and_notification_is_kept() -> None:
    guard, result, history, event = setup()
    observe(guard, result, .1, formula=True)
    observe(guard, result, .2)
    assert guard.notifications(0, event, history, None) == [event]


def test_score_increase_before_notification_is_kept() -> None:
    guard, result, history, event = setup()
    observe(guard, result, .1, score=140)
    observe(guard, result, .2, score=None)
    assert guard.notifications(0, event, history, None) == [event]


def test_formula_before_trigger_cannot_confirm() -> None:
    guard, result, history, event = setup()
    observe(guard, result, .05, formula=True)
    observe(guard, result, .2)
    assert not guard.notifications(0, event, history, None)


@pytest.mark.parametrize('name', ['overlay', 'replay', 'generate'])
def test_flag_defaults_off(name: str) -> None:
    import inspect
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    function = dict(overlay=ExchangeEventOverlay, replay=replay, generate=generate)[name]
    assert inspect.signature(function).parameters['death_formula_guard'].default is False
