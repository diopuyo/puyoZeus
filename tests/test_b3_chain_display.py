"""表示段は式の絶対段。初段欠落・途中分割・欠読・終了を検査する。"""
from __future__ import annotations

import pytest

from src.b3_chain_display import ChainDisplay, ScreenStage, formula_stage
from src.scoring import chain_power


def raw(number: int | None, state: str = 'CHAIN') -> dict:
    return dict(state=state, formula=dict(valid=number is not None,
        left=40 if number else None, right=max(1, chain_power(number)) if number else None))


@pytest.mark.parametrize('number', range(1, 20))
def test_absolute_stage(number: int) -> None:
    assert formula_stage(40, max(1, chain_power(number))) == number


def test_missing_first_and_false_end_do_not_count_from_one() -> None:
    tracker = ScreenStage()
    tracker.update(raw(5), 0.)
    tracker.update(raw(5), .03)
    assert tracker.value == 5
    tracker.update(raw(None, 'STABLE'), .1)
    tracker.update(raw(None, 'STABLE'), .3)
    assert tracker.value == 5
    tracker.update(raw(14), .4)
    tracker.update(raw(14), .43)
    assert tracker.value == 14


def test_missing_and_ambiguous_hold_then_quiet_end_clears() -> None:
    tracker = ScreenStage()
    tracker.update(raw(3), 0.)
    tracker.update(raw(3), .03)
    tracker.update(raw(None), .3)
    ambiguous = dict(state='CHAIN', formula=dict(valid=True, left=120, right=16))
    assert formula_stage(120, 16) is None
    tracker.update(ambiguous, .6)
    assert tracker.value == 3
    tracker.update(raw(None, 'STABLE'), 1.)
    tracker.update(raw(None, 'TSUMO_FALL'), 1.5)
    assert tracker.value is None


def test_single_read_is_not_adopted_and_new_chain_can_decrease() -> None:
    tracker = ScreenStage()
    tracker.update(raw(14), 0.)
    tracker.update(raw(14), .03)
    tracker.update(raw(9), .06)
    tracker.update(raw(None), .09)
    assert tracker.value == 14
    tracker.update(raw(1), .12)
    tracker.update(raw(1), .15)
    assert tracker.value == 1


def test_public_addition_preserves_feature_and_prediction() -> None:
    display = ChainDisplay()
    frame = dict(frame=0, t_sec=0., match_active=True, sides=[raw(14), raw(2)])
    display.observe(frame)
    display.observe(dict(frame, frame=2, t_sec=.03))
    contents = {side: dict(chain_n_steps=1, chain_pred_l=8) for side in ('p1', 'p2')}
    result = display.decorate(contents)
    assert result['p1']['chain_display_step'] == 14
    assert result['p2']['chain_display_step'] == 2
    assert result['p1']['chain_n_steps'] == 1
    assert result['p1']['chain_pred_l'] == 8
    assert 'chain_display_step' not in contents['p1']
    display.observe(dict(frame, frame=4, t_sec=.06, match_end_locked=True))
    assert display.decorate(contents)['p1']['chain_display_step'] is None


@pytest.mark.parametrize('interval', [.03, .1, .2, .5])
def test_live_cadence_does_not_require_thirty_fps(interval: float) -> None:
    tracker = ScreenStage()
    tracker.update(raw(5), 0.)
    tracker.update(raw(5), interval)
    assert tracker.value == 5


def test_internal_prediction_is_never_used_as_screen_count() -> None:
    tracker = ScreenStage()
    value = dict(raw(None), chain=dict(chain_count=19), formula_step=dict(observed_ordinal=7))
    tracker.update(value, 0.)
    tracker.update(value, .03)
    assert tracker.value is None
