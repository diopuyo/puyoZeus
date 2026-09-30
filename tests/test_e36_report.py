"""E36の契約(事前登録値・集計関数・起動条件)を固定する。"""
from pathlib import Path

from scripts import report_e36 as r


def gates(**kw):
    base = dict(q=dict(frames=6526, log_loss=.5080400023), zenchi=dict(frames=8333, hits=7671),
                deaths=dict(false=1, total=35, unlabelled=0),
                audit=dict(bound_false=0, unresolved=0), scene_sec=2760.1)
    base.update(kw)
    return r.evaluate_gates(base['q'], base['zenchi'], base['deaths'], base['audit'], base['scene_sec'])


def test_r1b_values_pass():
    assert all(gates().values())


def test_each_gate_can_fail():
    assert not gates(q=dict(frames=6526, log_loss=.5081))['q']
    assert not gates(q=dict(frames=6000, log_loss=.5))['q']
    assert not gates(zenchi=dict(frames=8333, hits=7670))['zenchi']
    assert not gates(deaths=dict(false=2, total=35, unlabelled=0))['false_fire']
    assert not gates(audit=dict(bound_false=1, unresolved=0))['false_certainty']
    assert not gates(audit=dict(bound_false=0, unresolved=1))['false_certainty']
    assert not gates(scene_sec=2766.1)['scene']
    assert not gates(scene_sec=None)['scene']


def test_route_and_risky_summary():
    dead = {'multi_landing': [{}, {'reason': 'already_dead'}]}
    assert r.classify_route(dead, '2P') == 'single'
    assert r.classify_route(dead, '1P') == 'other'
    bound = dict(dead, post_counter_bound=[{'dead': True, 'side': '2P'}])
    assert r.classify_route(bound, '2P') == 'other'
    case = dict(source='s', game=1, side='2P', first_sec=10., lead_sec=12., false=False)
    row = r.risky_row(case, dead, [dict(trigger=11., score=80)])
    out = r.summarize_risky([row, dict(row, false=True)])
    assert (out['cases'], out['single'], out['other_risky'], out['long_lead']) == (2, 2, 1, 2)


def test_contract_constants_and_launcher():
    assert r.OUT == Path('logs/e36') and r.RECORDS == Path('logs/r1b/records')
    assert len(r.SOURCES) == 5
    text = Path('scripts/_launch_e36.sh').read_text()
    assert 'nice -n 10' in text and 'scripts.run_e36' in text
