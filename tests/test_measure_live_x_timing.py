"""段0 測定台: 採点の数え方 (score_saved.py と同じ) と既知値の再現、M4・M5 の契約。"""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest

from scripts import measure_live_x_effects as effects
from scripts import measure_live_x_timing as timing

ORIGIN = 100.
OBS = Path('D:/puyo_analyzer/obs_vs_video_20261022')
REAL_DATA = timing.SAVED_21C.exists() and timing.ACTUAL.exists() and OBS.exists()
ALIGNED_21C = 57           # 21c 原票で発火が見つかる場面 (72 場面中)。2026-10-23 の再採点で確認


def scene(actual_score: int | None = 1000) -> dict:
    truth = None if actual_score is None else dict(comparable=True, chain=2, score=actual_score, accounting={})
    return dict(scene=dict(id=1, start=10., attacker=0, reply=11., attack=dict(end=15.)), X=truth)


def base(trigger: float) -> dict:
    return dict(predicted_scores=[[], []], chain_ids=[trigger, None])


def value(scores: list[int]) -> dict:
    return dict(predicted_scores=[scores, []], reconciliation=[dict(total=30), None],
                board_prediction_active=[True, False])


def write_run(root: Path, publications: list[tuple[float, float | None]],
              triggers: tuple[float, float] = (5., 10.)) -> Path:
    """(記録秒, v4.9f 完了ID) の公開票と、2つの v4.9f 結果 (発火 triggers) を持つ保存原票を作る。"""
    (root/'spool').mkdir(parents=True)
    with (root/'spool/evaluations.pickle').open('wb') as e, (root/'spool/publications.pickle').open('wb') as p:
        for frame, (stamp, completed) in enumerate(publications):
            row = dict(frame=frame, t_sec=stamp-.2, captured_at=stamp-.2+ORIGIN, evaluated_at=stamp+ORIGIN-.01,
                       v49f=dict(completed_at=completed))
            pickle.dump(row, e)
            pickle.dump(dict(frame=frame, evaluated_at=row['evaluated_at'], published_at=stamp+ORIGIN), p)
    updates = [dict(id=1, game=0, t=6., base=base(triggers[0]), value=value([300]), origins=[], completed_at=1.),
               dict(id=2, game=0, t=10.2, base=base(triggers[1]), value=value([400, 600]), origins=[],
                    completed_at=2.)]
    (root/'v49f_updates.jsonl').write_text(''.join(json.dumps(u)+'\n' for u in updates), encoding='utf-8')
    return root


def test_first_and_first_fresh_follow_saved_counting(tmp_path: Path) -> None:
    run = write_run(tmp_path, [(9.9, 2.), (10.1, 1.), (10.5, 2.), (17.1, 2.)])
    report = timing.score_saved_output(run, [scene()])
    row = report['rows'][0]
    assert row['first']['reason'] == '別の発火の保持値' and row['first']['public_sec'] == pytest.approx(10.1)
    assert row['first_fresh']['score'] == 1000 and row['first_fresh']['public_sec'] == pytest.approx(10.5)
    assert (report['M1_first_score'], report['M2_fresh_score'], report['first_present']) == (0, 1, 0)
    delay = report['M3']['rows'][0]
    assert delay['first_fresh_delay'] == pytest.approx(.5) and delay['input_delay'] == pytest.approx(.2)
    assert delay['publication_delay'] == pytest.approx(.3)


def aligned_scene(side: int, start: float, end: float = 15.) -> dict:
    return dict(scene=dict(id=start, start=start, attacker=side, reply=start+1., attack=dict(end=end)), X=None)


def test_early_fire_is_aligned_instead_of_dropped_as_held_value(tmp_path: Path) -> None:
    """走行の発火 (9.8) が通知時刻 (10.0) より早く出ても、同じ側 1.5 秒以内の最も近い発火なら当該 X。"""
    publications = [(9.9, 2.), (10.1, 1.), (10.5, 2.), (17.1, 2.)]
    run = write_run(tmp_path, publications, triggers=(5., 9.8))
    report = timing.score_saved_output(run, [scene()])
    row = report['rows'][0]
    assert row['first']['reason'] == '別の発火の保持値'              # 初回公開 (10.1) は 5.0 の発火の値
    assert row['first_fresh']['score'] == 1000 and row['first_fresh']['public_sec'] == pytest.approx(10.5)
    assert (report['M2_fresh_score'], report['M3']['n']) == (1, 1)
    assert report['alignment']['found'] == 1 and report['alignment']['offset_sec']['p50'] == pytest.approx(-.2)
    legacy = timing.score_saved_output(run, [scene()], align=False)      # 旧規則: 早い発火は捨てられる
    assert legacy['alignment'] is None
    assert (legacy['M2_fresh_score'], legacy['M3']['n']) == (0, 0)


def test_alignment_is_nearest_within_radius_unused_and_keeps_fragments() -> None:
    ids = [{9.5, 10.4, 12., 30.}, set()]
    scenes = [aligned_scene(0, 10.), aligned_scene(0, 10.5), aligned_scene(0, 11.)]
    aligned = timing.align_triggers(scenes, ids)
    first, second, third = (a['scene'] for a in aligned)
    assert first['_replay_trigger'] == 10.4 and first['_replay_triggers'] == [10.4, 12.]   # 最も近い (0.4 < 0.5)
    assert second['_replay_trigger'] == 9.5 and second['_replay_triggers'] == [9.5, 10.4, 12.]
    assert (third['_replay_trigger'], third['_replay_triggers']) == (12., [12.])   # 使用済みは再利用しない
    assert all('_replay_trigger' not in s['scene'] for s in scenes)               # 入力は変えない
    missing = timing.align_triggers([aligned_scene(1, 10.)], ids)[0]               # 別の側の発火は使わない
    assert missing['scene']['_replay_trigger'] == timing.NO_TRIGGER and missing['scene']['_replay_triggers'] == []
    summary = timing.alignment_summary(aligned+[missing])
    assert (summary['scenes'], summary['found']) == (4, 3) and summary['method'] == 'nearest_trigger'


def test_fire_beyond_radius_is_not_aligned(tmp_path: Path) -> None:
    """近傍 (1.5 秒) の外の発火は、同じ攻めの内でも対応付けない。見つからない場面は M3 の母数に入らない。"""
    assert timing.TRIGGER_RADIUS_SEC == 1.5
    late = [{11.6}, set()]
    assert timing.align_triggers([aligned_scene(0, 10.)], late)[0]['scene']['_replay_trigger'] == timing.NO_TRIGGER
    edge = [{11.5}, set()]
    assert timing.align_triggers([aligned_scene(0, 10.)], edge)[0]['scene']['_replay_trigger'] == 11.5
    run = write_run(tmp_path, [(10.1, 1.), (12.5, 2.)], triggers=(5., 12.))
    report = timing.score_saved_output(run, [scene()])
    assert (report['M3']['n'], report['alignment']['found']) == (0, 0)


def test_fire_ids_read_only_x_bases() -> None:
    updates = [dict(base=dict(predicted_scores=[[], []], chain_ids=[1., None])), dict(base=None),
               dict(base=dict(chain_ids=[9., 9.])), dict(value=dict()),
               dict(base=dict(predicted_scores=[[], []], chain_ids=[1., 2.]))]
    assert timing.fire_ids(updates) == [{1.}, {2.}]


def test_incomparable_truth_is_outside_denominator(tmp_path: Path) -> None:
    run = write_run(tmp_path, [(10.5, 2.)])
    report = timing.score_saved_output(run, [scene(None)])
    assert report['comparable'] == 0 and report['M3']['n'] == 0 and report['first_present'] == 1


def test_publication_placement_input_and_origin_guard(tmp_path: Path) -> None:
    run = write_run(tmp_path, [(10.5, 2.)])
    assert timing.load_publications(run, 'input')[0]['t'] == pytest.approx(10.3)
    with pytest.raises(ValueError, match='一定でない'):
        timing.run_origin([dict(captured_at=1., t_sec=0.), dict(captured_at=3., t_sec=1.)])


@pytest.mark.parametrize('trigger,extra,expected', [
    (10., {}, True), (15., {}, False), (9., {}, False),
    (12., {'_replay_trigger': 12.}, True), (10., {'_replay_trigger': 12.}, False),
    (13., {'_replay_triggers': [12., 13.]}, True), (None, {}, False)])
def test_valid_trigger(trigger: float | None, extra: dict, expected: bool) -> None:
    assert timing.valid_trigger(trigger, dict(scene()['scene'], **extra)) is expected


def test_jsonl_stops_at_partial_line(tmp_path: Path) -> None:
    path = tmp_path/'rows.jsonl'
    path.write_text('{"a": 1}\n{"a": 2}\n{"a"', encoding='utf-8')
    assert list(timing.jsonl_rows(path)) == [dict(a=1), dict(a=2)]


def test_known_check_reports_each_value() -> None:
    result = timing.check_known(dict(a=4, b=.7), dict(a=4, b=.71))
    assert not result['passed'] and result['checks']['a']['passed'] and not result['checks']['b']['passed']


def test_misfire_spans_and_withdrawal() -> None:
    rows = [dict(t=1., fires={0: 10.2}, provisional={}), dict(t=2., fires={0: 50.}, provisional={}),
            dict(t=3., fires={0: 50.}, provisional={}), dict(t=4., fires={}, provisional={}),
            dict(t=5., fires={1: 99.}, provisional={})]
    spans = effects.misfire_spans(rows, 'fires', {0: [10.], 1: []}, (0., 90.))
    assert spans == [dict(side=0, trigger=50., first_sec=2., withdrawn_sec=2.)]
    rows[3]['fires'] = {0: 50.}
    assert effects.misfire_spans(rows[:4], 'fires', {0: [10.], 1: []}, (0., 90.))[0]['withdrawn_sec'] is None


def test_semantic_drops_only_timing_columns(tmp_path: Path) -> None:
    row = dict(id=1, compute_ms=3., completed_at=9., ready_sec=1., over_budget=True, ms=2., monotonic=5.,
               value=dict(v_post=.5, low_spec_ms=4., input_sec=3.))
    assert effects.semantic(row) == dict(id=1, value=dict(v_post=.5, input_sec=3.))
    left, right = tmp_path/'l.jsonl', tmp_path/'r.jsonl'
    left.write_text(json.dumps(dict(game=0, id=1, v=1, compute_ms=1.))+'\n'+json.dumps(dict(game=0, id=2, v=2))+'\n')
    right.write_text(json.dumps(dict(game=0, id=1, v=1, compute_ms=9.))+'\n'+json.dumps(dict(game=0, id=2, v=3))+'\n')
    assert effects.compare_updates('v49f', left, right) == dict(left=2, right=2, common=2, different=1,
                                                               samples=['(0, 2)'])


def ama_row(index: int, cpu: float, setup: float, input_sec: float = 5., duration: float = 0.) -> dict:
    """ama 更新票の形: result.timings (走行ごとに変わる) と、同じ値になるべき _sec の列が混在する。"""
    return dict(generation=3, frame=100+index, item=dict(index=index, request=dict(input_sec=5., elapsed_sec=0.)),
                result=dict(input_sec=input_sec, cpu_sec=cpu,
                            timings=dict(setup_sec=setup, read_wall_sec=[.1, setup], read_cpu_sec=[.1, .2]),
                            value=dict(plans=[dict(automatic_fire=dict(duration_sec=duration))])))


def test_semantic_drops_ama_timings_and_cpu_sec_but_keeps_meaningful_sec_columns() -> None:
    """2026-10-23 是正: result.timings.*_sec と cpu_sec は除き、input_sec / elapsed_sec / duration_sec は残す。"""
    kept = effects.semantic(ama_row(1, cpu=9., setup=.9))
    assert 'timings' not in kept['result'] and 'cpu_sec' not in kept['result']
    assert kept['result']['input_sec'] == 5. and kept['item']['request'] == dict(input_sec=5., elapsed_sec=0.)
    assert kept['result']['value']['plans'][0]['automatic_fire'] == dict(duration_sec=0.)
    assert effects.semantic(ama_row(1, cpu=1., setup=.1)) == kept


def write_ama(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def test_compare_ama_ignores_timing_only_differences(tmp_path: Path) -> None:
    left, right = tmp_path/'l.jsonl', tmp_path/'r.jsonl'
    write_ama(left, [ama_row(i, cpu=1.+i, setup=.1) for i in range(3)])
    write_ama(right, [ama_row(i, cpu=7.+i, setup=.8) for i in range(3)])
    assert effects.compare_updates('ama', left, right) == dict(left=3, right=3, common=3, different=0, samples=[])
    # 意味のある _sec の列 (結果の input_sec・火力の duration_sec) の違いは、計時でないので不一致に数える。
    write_ama(right, [ama_row(i, cpu=7., setup=.8, input_sec=6. if i == 1 else 5., duration=1. if i == 2 else 0.)
                      for i in range(3)])
    result = effects.compare_updates('ama', left, right)
    assert (result['common'], result['different'], result['samples']) == (3, 2, ['(3, 101, 1)', '(3, 102, 2)'])


@pytest.mark.skipif(not REAL_DATA, reason='21c 保存原票が無い環境')
def test_known_values_are_reproduced_from_saved_originals() -> None:
    """原則3: 物差しはまず既知値 4/63・30/63・14/72・0.704/0.959 秒・全コマ 50/63・52/63 を再現する。"""
    scenes = timing.load_scenes()
    result = timing.reproduce_known(scenes)
    assert result['obs']['passed'] and result['full']['passed'], result
    assert (result['scenes'], result['comparable']) == (72, 63)
    report = timing.score_saved_output(timing.SAVED_21C, scenes)
    # 21c では発火 = 通知時刻なので、対応付けの是正で何も変わらない (57 場面で見つかり、ずれは全て 0)。
    assert report['alignment']['found'] == ALIGNED_21C and report['alignment']['offset_sec']['max'] == 0.
    assert timing.score_saved_output(timing.SAVED_21C, scenes, align=False)['rows'] == report['rows']
    saved = {r['id']: r for r in json.loads((OBS/'saved_X_rows.json').read_text(encoding='utf-8'))}
    for row in report['rows']:
        expected = saved[row['id']]
        for key in ('first', 'first_fresh'):
            assert row[key].get('score') == expected[key].get('score')
            assert row[key].get('public_sec') == expected[key].get('public_sec')
    delays = {r['id']: r for r in json.loads((OBS/'first_fresh_delay.json').read_text(encoding='utf-8'))['rows']}
    assert {r['id']: r for r in report['M3']['rows']} == delays
