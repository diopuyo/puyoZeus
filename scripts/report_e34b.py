"""新しい同一入力上でE32/E34を採点し、旧入力との差を別集計する。"""
from __future__ import annotations

from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

from scripts import e33b_score_trace as score_trace
from scripts import report_e33b as scores_module
from scripts import run_e17_ablation_20260928 as prior
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json, digest
from scripts.report_e33b import read
from src.exchange_event_record import read_records

OUT = Path('logs/e34b')
LOSS_ALLOWANCE, AGREEMENT_ALLOWANCE, SCORE_RATIO = .002, .003, 1.05


def directory(variant: str, source: str) -> Path:
    """既存集計器と同じ出力構造を使う。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    return OUT/variant/suffix


def updates(path: Path) -> Any:
    """静止特徴などを除き、認識更新の全行を逐次読む。"""
    return (r['args'] for r in read_records(path) if r['kind'] == 'update')


def input_difference(source: str) -> dict:
    """同じ時刻の確定盤面を比較し、Noneと実盤面の違いも不一致に含める。"""
    rows, mismatch, sides, first_sec = 0, 0, [0, 0], None
    left = updates(Path('logs/e31/records')/f'{source}.jsonl.gz')
    right = updates(OUT/'records'/f'{source}.jsonl.gz')
    first_side: list[float | None] = [None, None]
    for before, after in zip_longest(left, right):
        assert before is not None and after is not None, source
        assert before[3] == after[3], (source, before[3], after[3])
        changed = []
        for idx, label in enumerate(('p1', 'p2')):
            a, b = (getattr(value[0], label).confirmed_board for value in (before, after))
            equal = (a is None and b is None) or (a is not None and b is not None and
                    np.array_equal(a._grid, b._grid))
            changed.append(not equal)
            sides[idx] += not equal
            if not equal and first_side[idx] is None:
                first_side[idx] = before[3]
        rows += 1
        mismatch += any(changed)
        if any(changed) and first_sec is None:
            first_sec = before[3]
    return dict(rows=rows, board_mismatch_rows=mismatch, side_mismatch_rows=sides,
                first_mismatch_sec=first_sec, first_side_mismatch_sec=first_side)


def prepare_cohort() -> dict:
    """ON実行前に新OFFの採用連鎖・真値・表示行をE33bの方式で固定する。"""
    score_trace.OUT = OUT
    score_trace.baseline_directory = lambda s: directory('off', s)
    result = score_trace.prepare_cohort()
    save_json(OUT/'NONDETERMINISM.json', {s: input_difference(s) for s in prior.ALL_SOURCES})
    for source in prior.ALL_SOURCES:
        path = OUT/'records'/f'{source}.jsonl.json'
        windows = OUT/'records'/f'{source}.windows.json.gz'
        save_json(path, dict(read(path), windows_path=str(windows), windows_sha256=digest(windows)))
    return result


def compare_rows() -> dict:
    """表示時刻と現在層の入力列を全行比較し、母数を共通部分へ縮めない。"""
    result = {}
    for source in prior.ALL_SOURCES:
        with np.load(directory('off', source)/'display.npz') as a, np.load(
                directory('on', source)/'display.npz') as b:
            assert a.files == b.files
            for name in a.files:
                if name not in ('display_p1', 'display_adv', 'source'):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=f'{source}:{name}')
            result[source] = dict(rows=len(a['t_sec']), columns=len(a.files))
    return result


def adoptions(variant: str) -> dict:
    """3動画を母集団とし、重複区間のreviewとzenchiを加算しない。"""
    rows = [r for s in SOURCES for r in read(directory(variant, s)/'snapshot_final_audit.json')['rows']]
    return dict(fires=len(rows), accepted=sum(r['accepted'] for r in rows),
                rejected=Counter(r['reason'] for r in rows if not r['accepted']))


def metrics(variant: str) -> dict:
    """qの独立勝敗・zenchi公式勝敗・誤発火の既存採点を再利用する。"""
    prior.OUT = OUT
    value = prior.report(variant)
    save_json(OUT/f'DEATH_AUDIT_{variant}.json', prior.report_base.death_metrics.deaths())
    value['scene_first_sec'] = first(*displayed(directory(variant, 'review')/'display.npz'))
    return value


def report() -> dict:
    """事前登録した相対許容幅を、新OFFの実測値に適用する。"""
    checks = compare_rows()
    values = {v: metrics(v) for v in ('off', 'on')}
    scores_module.OUT, scores_module.VARIANTS = OUT, ('off', 'on')
    scores = scores_module.scores()
    a, b = values['off'], values['on']
    assert a['q']['frames'] == b['q']['frames'] and a['zenchi']['frames'] == b['zenchi']['frames']
    limits = dict(q=a['q']['log_loss']+LOSS_ALLOWANCE,
                  zenchi=a['zenchi']['agreement']-AGREEMENT_ALLOWANCE, score=scores['off']['mean']*SCORE_RATIO)
    gates = dict(q=b['q']['log_loss'] <= limits['q'], zenchi=b['zenchi']['agreement'] >= limits['zenchi'],
        deaths=b['deaths']['unlabelled'] == 0 and b['deaths']['false']/max(1, b['deaths']['total']) <= 1/28,
        score_mean=scores['on']['mean'] <= limits['score'])
    for variant, value in values.items():
        value.pop('gates', None)
        value.pop('candidate', None)
        save_json(OUT/variant/'METRICS.json', value)
    result = dict(metrics=values, scores=scores, limits=limits, gates=gates, passed=all(gates.values()),
        comparisons=checks, nondeterminism=read(OUT/'NONDETERMINISM.json'),
        cohort=read(OUT/'COHORT.json'), adoptions={v: adoptions(v) for v in ('off', 'on')})
    save_json(OUT/'SUMMARY.json', result)
    return result


if __name__ == '__main__':
    print(report())
