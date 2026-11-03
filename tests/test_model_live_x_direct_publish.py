"""段1 の見込み再生器: 待ち行列・起床・判定IDの重複排除の規則と、21c 既知値の再現。"""
from __future__ import annotations

import pytest

from scripts import measure_live_x_timing as timing
from scripts import model_live_x_direct_publish as model

KEY = (1, 1, 'static', .5, True, 0)


def result(frame: int, completed: float, compute_ms: float, wait_ms: float = 0., x: bool = True) -> dict:
    base = dict(predicted_scores=[[], []], chain_ids=[.96 if x else None, None])
    value = dict(predicted_scores=[[40, 60] if x else [], []], reconciliation=[None, None],
                 board_prediction_active=[True, False])
    return dict(id=frame, game=0, t=.9, completed_at=completed, compute_ms=compute_ms, wait_ms=wait_ms,
                base=base, value=value, origins=[])


def evaluation(frame: int, stamp: float, key: tuple = KEY, completed: float | None = None) -> dict:
    return dict(frame=frame, evaluated_at=stamp, judgment_id=key, game=0, v49f=dict(completed_at=completed))


def dataset() -> dict:
    evaluations = [evaluation(9, .9), evaluation(11, 1.3, completed=1.), evaluation(12, 1.6, (1, 2, 'static', .6, True, 0), 1.)]
    publications = [dict(frame=9, evaluated_at=.9, published_at=.9005), dict(frame=12, evaluated_at=1.6, published_at=1.6005)]
    return dict(results=[result(9, 1., 500.)], ready={9: .5}, evaluations=evaluations,
                busy=[(.85, .95), (1.25, 1.35), (1.55, 1.65)], publications=publications, origin=0.)


def scenes() -> list[dict]:
    return [dict(scene=dict(id=1, start=.95, attacker=0, reply=1., attack=dict(end=2.)),
                 X=dict(comparable=True, chain=2, score=100, accounting={}))]


def test_worker_fifo_without_pump_round_trip() -> None:
    results = [result(1, 1., 500.), result(2, 1.2, 100.)]
    assert model.worker_timeline(results, {1: .5, 2: .6}, False) == [1., 1.2]
    assert model.worker_timeline(results, {1: .5, 2: .6}, True) == pytest.approx([1., 1.1])


def test_wake_waits_for_the_notice_in_progress() -> None:
    busy = [(1., 1.05), (2., 2.3)]
    starts = [b[0] for b in busy]
    assert model.wake_time(busy, starts, 2.1, .002) == pytest.approx(2.302)
    assert model.wake_time(busy, starts, 1.5, .002) == pytest.approx(1.502)


def test_cursor_drops_other_identity_and_resets() -> None:
    results = [result(9, 1., 1.), result(10, 2., 1.)]
    cursor = model.ResultCursor(results, [1., 2.], [True, False], {9: 1, 10: 2})
    cursor.advance(1.5, (1, 0))
    assert (cursor.latest, cursor.x_done) == (1., 1.)
    cursor.advance(2.5, (1, 0))
    assert cursor.latest == 1.
    cursor.advance(3., (2, 0))
    assert (cursor.latest, cursor.x_done) == (None, None)


def test_off_publishes_only_on_judgment_change_and_on_wakes_for_x() -> None:
    data = dataset()
    off = model.simulate(data, False)
    assert [round(r['t'], 4) for r in off['display']] == [.9005, 1.6005]
    assert model.compare_actual(off, data) == dict(simulated=2, actual=2, both=2, only_simulated=0, only_actual=0)
    on = model.simulate(data, True, .002)
    assert [round(r['t'], 4) for r in on['display']] == [.9005, 1.0025, 1.6005]
    assert on['display'][1]['vf']['completed_at'] == 1. and on['x_first'] == 1
    delays = [timing.score_run(s['display'], timing.index_results(data['results']), scenes())['M3']
              for s in (off, on)]
    assert delays[0]['rows'][0]['first_fresh_delay'] == pytest.approx(.6505)
    assert delays[1]['rows'][0]['first_fresh_delay'] == pytest.approx(.0525)


@pytest.mark.skipif(not timing.SAVED_21C.exists(), reason='21c 保存原票が無い環境')
def test_off_replay_reproduces_21c_and_on_keeps_values() -> None:
    """原則3: 再生器は OFF で 21c の既知値を全部再現してから、ON の見込みに使う。"""
    result = model.estimate(timing.SAVED_21C, timing.load_scenes(), (2.,))
    assert result['off_known']['passed'], result['off_known']
    assert result['off']['vf_agreement'] > .99
    on = result['on']['trigger_2ms']
    assert (on['M1_first_score'], on['M2_fresh_score'], on['fresh_n']) == (4, 30, 51)
    assert on['fresh_p50'] < result['off']['fresh_p50'] and on['fresh_p95'] <= result['off']['fresh_p95']
