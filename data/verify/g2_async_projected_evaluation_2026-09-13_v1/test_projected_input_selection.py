"""新DTO入口の欠測/起点矛盾分岐だけを人工入力で検査する。物理数値は実replayで別検査。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as N
from typing import Any
import pytest
import prefix_projected_input as P


def fixture() -> tuple:
    grid = tuple((0,) * 6 for _ in range(13))
    encoded = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    board = dict(grid=grid, sha256=hashlib.sha256(encoded(grid).encode()).hexdigest())
    origin = dict(object_id=1, trigger_sec=1.0, before_board=board)
    step = dict(returned=dict(active_origin=origin), events=[dict(active_origin=deepcopy(origin))])
    family = N(value=N(frame=30, worlds=(N(grid=grid),)))
    lane = N(families=(family,))
    source = N(MODE_KEY='mode', read=lambda *args: (lane, step, None))
    belief = N(HIDDEN_ROWS=1, Board=N(from_dict=lambda value: value), grid=lambda value: value['grid'])
    live = N(module=N(B=belief), parts=object())
    lease = N(current=lambda: (live, None))
    session = N(state={'mode': N(arrival_ledger=object())})
    def unexpected(*args: Any) -> None:
        raise AssertionError('欠測/矛盾入力で物理投影へ到達した')
    return session, lease, source, N(project=unexpected, encoded=encoded), step, lane


@pytest.mark.parametrize('fault,reason', [
    ('source', 'artificial_source_hold'), ('missing', 'returned_origin_not_available'),
    ('settled', 'returned_origin_is_settled_notice'), ('board', 'returned_origin_board_not_available'),
    ('events', 'returned_origin_not_observed_in_call'),
    ('old_origin', 'returned_origin_not_represented_by_family'),
    ('different_grid', 'returned_origin_not_represented_by_family')])
def test_missing_input_never_becomes_prediction(fault: str, reason: str) -> None:
    session, lease, source, projection, step, lane = fixture()
    if fault == 'source': source.read = lambda *args: (None, None, reason)
    elif fault == 'missing': step['returned']['active_origin'] = None
    elif fault == 'board': step['returned']['active_origin']['before_board'] = None
    elif fault == 'events': step['events'] = []
    elif fault == 'old_origin': lane.families[0].value.frame = 61
    elif fault == 'different_grid': lane.families[0].value.worlds = (N(grid=((1,) * 6,) * 13),)
    result = P.read(session, lease, None, source, projection, 100, settled=lambda value: fault == 'settled')
    assert result['status'] == 'HOLD' and result['reason'] == reason and result['prediction'] is None
    assert not result['quality_gate_clear'] and not result['source_producer_authorized']


@pytest.mark.parametrize('key', ['trigger_sec', 'before_board'])
def test_conflicting_raw_origin_is_strict(key: str) -> None:
    session, lease, source, projection, step, _ = fixture()
    step['events'][0]['active_origin'][key] = None
    with pytest.raises(ValueError, match='origin_mutated'):
        P.read(session, lease, None, source, projection, 100, settled=lambda value: False)


def test_shared_corrupt_sha_is_not_accepted_as_agreement() -> None:
    session, lease, source, projection, step, _ = fixture()
    step['returned']['active_origin']['before_board']['sha256'] = 'corrupt'
    step['events'][0]['active_origin']['before_board']['sha256'] = 'corrupt'
    with pytest.raises(ValueError, match='source_board_hash'):
        P.read(session, lease, None, source, projection, 100, settled=lambda value: False)
