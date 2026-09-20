"""実保存由来の陰性を使う。元入力と原実装は変更しない。"""
from __future__ import annotations
from copy import deepcopy
from typing import Any
import pytest
import run_world as R
import world_verify as W

FIRING_FRAME, PREFIX_FRAME, NEXT_FRAME = 34910, 34866, 34934


@pytest.fixture(scope='module')
def saved() -> Any:
    return R.inputs()


def at(value: Any, frame: int) -> Any:
    return next(row for row in value['history_rows'] if row['scope']['frame_idx'] == frame)


def test_original_world(saved: Any) -> None:
    value = W.verify_world(**saved)
    assert value['world_PB_verified'] and value['prepared_worlds'] == 5
    assert not value['runtime_finalization_allowed'] and not value['actual_live_scope_verified']


@pytest.mark.parametrize('case', ('prefix_pair', 'prefix_support', 'tail_source', 'next_pair',
    'prior_certificate', 'origin_backend', 'origin_prediction', 'origin_erased', 'origin_capture',
    'settlement_capture', 'settlement_duplicate', 'hidden_history_missing', 'hidden_history_duplicate',
    'hidden_source', 'lifetime_anchor', 'lifetime_pending', 'lifetime_duplicate'))
def test_world_tamper(saved: Any, case: str) -> None:
    value = deepcopy(saved)
    origin = next(r for r in value['conditional_rows'] if r.get('stage') == 'conditional_origin_registered')
    settle = next(r for r in value['conditional_rows'] if r.get('stage') == 'conditional_private_settled')
    if case == 'prefix_pair': at(value, PREFIX_FRAME)['prepared']['pair'] = [1, 1]
    elif case == 'prefix_support': at(value, PREFIX_FRAME)['prepared']['inferred_path']['final'][0][0] = 1
    elif case == 'tail_source': at(value, 34870)['prepared']['prefix_source']['old_token'] += ':foreign'
    elif case == 'next_pair': at(value, NEXT_FRAME)['prepared']['pair'] = [1, 1]
    elif case == 'prior_certificate': at(value, NEXT_FRAME)['prepared']['previous_conditional_certificate'] = '{}'
    elif case == 'origin_backend': origin['proof']['backend']['adopted_ghost_rule'] = False
    elif case == 'origin_prediction': origin['proof']['placement']['predicted_final'][0][0] = 1
    elif case == 'origin_erased':
        origin['origin']['erased'][0] += 1
        origin['proof']['evidence']['erased'][0] += 1
    elif case == 'origin_capture': origin['proof']['placement']['observations'][0]['raw_proof']['captured_frame'] -= 2
    elif case == 'settlement_capture': settle['proof']['observations'][0]['raw_capture']['captured_frame'] -= 2
    elif case == 'settlement_duplicate': value['conditional_rows'].append(deepcopy(settle))
    elif case == 'hidden_history_missing': value['hidden_history'].pop()
    elif case == 'hidden_history_duplicate': value['hidden_history'].append(deepcopy(value['hidden_history'][0]))
    elif case == 'hidden_source': value['hidden_history'][0]['source']['token'] += ':foreign'
    elif case == 'lifetime_anchor': value['hidden_lifetime'][0]['anchor_retained'] = False
    elif case == 'lifetime_pending': value['hidden_lifetime'][0]['pending'] = True
    elif case == 'lifetime_duplicate': value['hidden_lifetime'].append(deepcopy(value['hidden_lifetime'][0]))
    with pytest.raises((ValueError, AssertionError)):
        W.verify_world(**value)
