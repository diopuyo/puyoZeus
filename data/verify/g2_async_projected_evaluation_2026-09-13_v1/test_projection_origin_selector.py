"""元next選択の曖昧性を再現し、対象pinと正常対照を検査する。"""
from copy import deepcopy
import pytest
from projection_origin_selector import selected_origin


def fixture() -> dict:
    origin = dict(object_id=123, trigger_sec=10.0, before_board={'grid': [[1]]})
    return dict(side='1P', token='step:1', events=[{'active_origin': origin}])


def test_original_first_match_hides_ambiguity() -> None:
    step = fixture()
    step['events'].append({'active_origin': dict(object_id=456, trigger_sec=10.0,
                                               before_board={'grid': [[1]]})})
    assert next(e['active_origin'] for e in step['events'])['object_id'] == 123
    with pytest.raises(ValueError, match='projection_target_origin'):
        selected_origin(step, '1P', 'step:1', 123)


@pytest.mark.parametrize('field,value', [('side', '2P'), ('token', 'step:2')])
def test_wrong_call_rejected(field: str, value: str) -> None:
    step = fixture()
    step[field] = value
    with pytest.raises(ValueError, match='projection_target_call'):
        selected_origin(step, '1P', 'step:1', 123)


def test_normal_repeated_stage_and_mutation() -> None:
    step = fixture()
    step['events'].append(deepcopy(step['events'][0]))
    before = deepcopy(step)
    assert selected_origin(step, '1P', 'step:1', 123) == step['events'][0]['active_origin']
    assert step == before
    step['events'][1]['active_origin']['before_board']['grid'] = [[2]]
    with pytest.raises(ValueError, match='projection_origin_mutated'):
        selected_origin(step, '1P', 'step:1', 123)
