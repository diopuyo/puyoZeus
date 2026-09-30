"""R1bの遅延比較でOFFも同じ母数に揃えることを検証する。"""
from __future__ import annotations
import pytest
import json
from pathlib import Path
from scripts.measure_r1b import common_delays
from scripts.report_r1b import counter, SCENE_TIME


def row(index: int, old: float | None, new: float | None) -> dict:
    """同一置きの固定接地基準と反映差を表す最小原票。"""
    return dict(source='test', t_sec=float(index), side=0, old_delay=old, new_delay=new)


def test_off_quantiles_use_the_same_intersection() -> None:
    previous = [row(0, 1., 1.), row(1, 9., 9.), row(2, 100., None), row(3, 200., 200.)]
    current = [row(0, 1., 1.), row(1, 9., 9.), row(2, 100., 100.), row(3, 200., None)]
    value = common_delays(previous, current)
    assert value['passed'] and value['indices'] == [0, 1]
    assert value['paired'] == 2 and value['total'] == 4
    assert value['missing'] == dict(off=0, r1=1, r1b=1)
    assert value['quantiles']['off'] == value['quantiles']['r1'] == value['quantiles']['r1b']
    assert value['quantiles']['off']['p50'] == 5.


def test_changed_delay_fails_on_the_same_cohort() -> None:
    value = common_delays([row(0, 1., 1.)], [row(0, 1., 2.)])
    assert not value['passed'] and value['paired'] == 1


def test_no_paired_placements_is_not_a_pass() -> None:
    value = common_delays([row(0, 1., None)], [row(0, 1., 1.)])
    assert not value['passed'] and value['paired'] == 0


@pytest.mark.parametrize('field,value', [('source', 'other'), ('old_delay', 2.)])
def test_changed_cohort_or_contact_anchor_is_rejected(field: str, value: object) -> None:
    current = row(0, 1., 1.)
    current[field] = value
    with pytest.raises(AssertionError):
        common_delays([row(0, 1., 1.)], [current])


def test_counter_uses_last_past_evaluation_not_future(tmp_path: Path) -> None:
    dest = tmp_path/'on/review'
    dest.mkdir(parents=True)
    prediction = dict(side='2P', mean_score=75440., mean_send=1077.)
    values = [dict(source='S3_landing', t_sec=SCENE_TIME-1, prefire_prediction=[prediction]),
              dict(source='S3_landing', t_sec=SCENE_TIME+1, prefire_prediction=[])]
    (dest/'events.jsonl').write_text(json.dumps(dict(values=values))+'\n')
    value = counter(tmp_path, 'on')
    assert value['adopted'] and value['predictions'] == [prediction]
    assert value['t_sec'] == SCENE_TIME-1
