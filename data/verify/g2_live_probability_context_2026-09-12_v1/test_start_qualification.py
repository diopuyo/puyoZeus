"""開始資格の因果性/得点保持/拒否条件。全入力は人工で、実動画品質ではない。"""
from __future__ import annotations
from copy import deepcopy
import pytest
import start_qualification as Q

FIRST, RESET, OCCURRENCE, QUALIFIED, LAST = 100, 102, 106, 136, 140
DEBOUNCE = 5.0  # 凍結collectorの境界値と一致する人工対照。


def example() -> dict:
    rows = []
    for frame in range(FIRST, LAST + Q.STRIDE, Q.STRIDE):
        menu = RESET <= frame < OCCURRENCE
        rows.append(dict(frame=frame, game=int(frame >= QUALIFIED), active=frame >= OCCURRENCE,
            gates=[True, True], states=['menu' if menu else 'tsumo_fall'] * Q.SIDES,
            scores=[6, 6], resets=[int(frame >= RESET)] * Q.SIDES,
            activity={name: [0, 0] for name in Q.ACTIVITY},
            pending=[0, 0], capped=[0, 0], leftover=[0, 0], unsettled=[False, False]))
    events = [dict(list=name, raw_value=OCCURRENCE / Q.FPS,
                   available_frame=QUALIFIED, available_sec=QUALIFIED / Q.FPS)
              for name in ('advance_times', 'visual_rise_times')]
    return dict(identity=dict(source_id='artificial-video', run_id='artificial-run'),
                first_frame=FIRST, last_frame=LAST, start_counter_trace=rows, boundary_events=events)


def decide(value: dict) -> dict:
    return Q.decide(value, debounce_sec=DEBOUNCE)


def test_score_six_and_missing_board_are_not_rewritten() -> None:
    value = example()
    before = deepcopy(value)
    result = decide(value)
    assert result['eligible'] and result['evidence']['scores'] == [6, 6]
    assert result['evidence']['reset_frames'] == [RESET, RESET]
    assert value == before


def test_future_activity_does_not_change_start_evidence() -> None:
    value = example()
    expected = decide(value)
    for row in value['start_counter_trace']:
        if row['frame'] > QUALIFIED:
            row['activity']['generated'] = [1, 0]
            row['activity']['finalized'] = [1, 0]
            row['pending'] = [0, 1]
    assert decide(value) == expected


@pytest.mark.parametrize('side', [0, 1])
def test_single_side_reset_is_not_qualified(side: int) -> None:
    value = example()
    for row in value['start_counter_trace']:
        row['resets'][side] = 0
    assert decide(value)['reason'] == 'observed_both_menu_resets_missing'


def test_constructor_zero_is_not_observed_reset() -> None:
    value = example()
    for row in value['start_counter_trace']:
        row['resets'] = [0, 0]
    assert not decide(value)['eligible']


def test_score_drop_reset_without_menu_is_not_qualified() -> None:
    value = example()
    for row in value['start_counter_trace']:
        row['states'] = ['stable', 'stable']
    assert decide(value)['reason'] == 'nonmenu_reset_in_window'


@pytest.mark.parametrize('name', Q.ACTIVITY)
def test_activity_before_start_rejects(name: str) -> None:
    value = example()
    for row in value['start_counter_trace'][1:]:
        row['activity'][name] = [1, 0]
    assert decide(value)['reason'] == 'activity_since_reset'


def test_late_visual_with_debounced_advance_is_explicit() -> None:
    value = example()
    value['boundary_events'][0].update(raw_value=RESET / Q.FPS, available_frame=RESET,
                                       available_sec=RESET / Q.FPS)
    for row in value['start_counter_trace'][1:]:
        row['game'] = 1
    assert decide(value)['eligible']


def test_next_game_expires_old_start() -> None:
    value = example()
    value['start_counter_trace'][-1]['game'] = 2
    assert decide(value)['reason'] == 'start_game_expired'


def test_review125_menu_at_qualification_is_not_accepted() -> None:
    value = example()
    for row in value['start_counter_trace']:
        if row['frame'] == QUALIFIED:
            row['states'] = ['menu', 'menu']
    assert not decide(value)['eligible']


def test_review125_short_visual_persistence_is_not_accepted() -> None:
    value = example()
    for event in value['boundary_events']:
        event['raw_value'] = (QUALIFIED - Q.STRIDE) / Q.FPS
    assert not decide(value)['eligible']


def test_reset_edge_mismatch_has_distinct_reason() -> None:
    value = example()
    value['start_counter_trace'][0]['states'] = ['menu', 'menu']
    assert decide(value)['reason'] == 'menu_reset_edge_mismatch'


def test_later_nonmenu_reset_invalidates_with_explicit_reason() -> None:
    value = example()
    for row in value['start_counter_trace']:
        if row['frame'] >= OCCURRENCE:
            row['resets'][0] = 2
    assert decide(value)['reason'] == 'nonmenu_reset_in_window'


def test_frozen_configuration_mismatch_is_integrity_error() -> None:
    with pytest.raises(ValueError, match='debounce'):
        Q.decide(example(), debounce_sec=Q.DEBOUNCE_SEC - 1)


@pytest.mark.parametrize('mutation,reason', [('missing', 'coverage'), ('counter', 'activity_regressed'),
                                          ('future', 'visual_available_frame')])
def test_malformed_evidence_raises(mutation: str, reason: str) -> None:
    value = example()
    if mutation == 'missing':
        del value['start_counter_trace'][1]
    elif mutation == 'counter':
        value['start_counter_trace'][0]['activity']['generated'] = [1, 0]
    else:
        value['boundary_events'][-1]['available_frame'] = LAST + Q.STRIDE
    with pytest.raises(ValueError, match=reason):
        decide(value)
