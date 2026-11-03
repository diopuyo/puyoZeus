"""残存診断の全表・照合結果・ファイル一覧を指定先に新規作成する。"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts._diag_remaining_20261002 import BASE, OUT, ROOT, MIN_MEMORY, read, save

MODES = ('all', 'prod', 'no_d3')


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Markdown全表を組み立てる。"""
    def cell(value: Any) -> str:
        return '—' if value is None else str(value).replace('|', '/')
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    return '\n'.join(lines+['| '+' | '.join(map(cell, row))+' |' for row in rows])+'\n'


def stamp(value: float | None) -> str:
    """秒値は表示のみ小数3桁へ丸める。"""
    return '—' if value is None else f'{value:.3f}'


def traces(source: str, mode: str) -> list[dict]:
    """観測原票を読む。"""
    path = OUT / f'trace_{source}_{mode}/trace.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def inputs() -> tuple[list, list]:
    """確定時点の生入力と実際の死亡用入力を混同せずに保存する。"""
    rows, boards, evidence = [], [], []
    for case in read(OUT / 'false_cases.json'):
        source, when = case['source'], case['first_sec']
        for mode in MODES if source != 'c5' else MODES[:2]:
            values = traces(source, mode)
            update = next(r for r in values if r['kind']=='update' and abs(r['t_sec']-when)<.001)
            death = next((r for r in values if r['kind']=='death_inputs' and abs(r['t_sec']-when)<.001), None)
            sides = update['args'][0]
            evidence.append(dict(case=case, mode=mode, update=update, death=death))
            for idx, key in enumerate(('p1', 'p2')):
                side = sides[key]
                latest = death['latest'][idx] if death else None
                rows.append([source, case['game'], mode, key.upper(), side['state']['_name_'],
                    side['score'], update['args'][6][idx], side['next_pair'], side['dnext_pair'],
                    side['next_slide_motion'], stamp(latest['t_sec']) if latest else None,
                    death['context']['incoming'][idx] if death else None,
                    death['context']['verified'][idx] if death else None])
                if mode == 'all':
                    grid = death['context']['replies'][idx]
                    encoded = '/'.join(''.join(str(c) for c in row) for row in grid)
                    boards.append([source, case['game'], key.upper(), encoded])
    save('input_evidence.json', evidence)
    return rows, boards


def verify() -> list[list]:
    """再生表示の全列を、既存成果物の同時刻区間と数値照合する。"""
    checks = []
    for directory in sorted(OUT.glob('trace_*')):
        source, mode = directory.name.removeprefix('trace_').rsplit('_', 1)
        if mode == 'd3':
            source, mode = source.removesuffix('_no'), 'no_d3'
        with np.load(directory/'display.npz') as actual, np.load(BASE/f'replay/{mode}/{source}/display.npz') as expected:
            n = len(actual['t_sec'])
            for key in actual.files:
                if actual[key].ndim == 1 and len(actual[key]) == n:
                    np.testing.assert_array_equal(actual[key], expected[key][:n], err_msg=f'{directory.name}:{key}')
            checks.append([source, mode, n, '全表示列一致' if n else '表示なし・イベント照合'])
        for row in traces(source, mode):
            if row['kind'] != 'update':
                continue
            if source == 'c5':
                checks[-1][-1] = '対象時刻の評価状態を記録（表示区間外）'
    save('validation.json', checks)
    return checks


def loss_tables(result: dict) -> tuple[list, list, list]:
    """消失全52事例と重複あり帰属、観測死亡までの分位を作る。"""
    rows, aggregate, distribution = [], [], []
    lost = read(OUT/'lost_attribution.json')
    for case in lost:
        restored = case['restored']
        rows.append([case['set'], case['game'], case['side'], stamp(case['first_sec']),
                     stamp(case['candidate_observed_death_sec']), stamp(case['delay_sec']),
                     *[stamp(restored[m]) for m in ('no_d1','no_d2','no_d3','no_d4')]])
    for name, data in result['all']['sets'].items():
        selected = [r for r in lost if r['set']==name]
        counts = [sum(r['restored'][f'no_d{i}'] is not None for r in selected) for i in range(1,5)]
        aggregate.append([name, data['games'], data['deaths']['correct_before'], len(selected), *counts,
                          sum(all(v is None for v in r['restored'].values()) for r in selected)])
        quantiles = data['correct_delay']['lost_delay']['quantiles']
        distribution.append([name,len(selected),0,*[stamp(quantiles[k]) for k in ('min','p25','p50','p75','p95','max')]])
    return rows, aggregate, distribution


def scene() -> tuple[list, list]:
    """実呼出の拒否理由と回帰値の定義を記録する。"""
    values = [r for r in traces('review','all') if r['kind']=='post_counter_call']
    counts = Counter(r['result']['reason'] for r in values)
    rows = [[key,count,stamp(min(r['t_sec'] for r in values if r['result']['reason']==key)),
             stamp(max(r['t_sec'] for r in values if r['result']['reason']==key))] for key,count in counts.items()]
    data = read(BASE/'RESULT.json')
    times = [[mode, stamp(data[mode]['regression']['scene']['first_display_sec']),
              stamp(data[mode]['regression']['scene']['legacy_metric_sec'])] for mode in data]
    save('scene_review_counts.json', dict(source='review', interval=[2760,2772], calls=len(values),
        counts=dict(counts), rows=rows, note='最初に該当した拒否理由。未到達の条件を0回と断定しない。'))
    return rows, times


def main() -> None:
    """最終レポートの全表を作成する。"""
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(memory['MemAvailable'].split()[0])*1024 >= MIN_MEMORY
    result = read(BASE/'RESULT.json')
    raw, boards = inputs()
    checks = verify()
    lost, aggregate, distribution = loss_tables(result)
    reasons, times = scene()
    sections = [('入力（生得点・表示得点、NEXT、死亡用台帳）',
        ['記録','試合','構成','側','状態','side.score','表示OCR','NEXT','NEXT2','移動','採用盤面時刻','死亡用予告','verified'],raw),
        ('死亡証明へ渡した盤面（row0→12、各行左→右、/は行区切り）',['記録','試合','側','盤面'],boards),
        ('消失事例の外し比較（復活時刻、—は復活なし）',
         ['セット','試合','側','prod確定','観測死亡','遅れ秒','D1外し','D2外し','D3外し','D4外し'],lost),
        ('帰属集計（外し復活は重複あり）',['セット','全試合','prod正確定','消失','D1','D2','D3','D4','一つ抜きで未同定'],aggregate),
        ('観測死亡までの遅れ分布（秒）',['セット','件数','欠測','最小','p25','中央値','p75','p95','最大'],distribution),
        ('review 2760〜2772秒：E35呼出86回の拒否',['拒否理由','回数','最初','最後'],reasons),
        ('3:00回帰の時刻（死亡確定時刻ではなくP1確率0.95到達）',['構成','直接表示秒','旧EMA再適用秒'],times),
        ('観測再生と既存成果物の照合',['記録','構成','表示フレーム数','照合'],checks)]
    report = '\n\n'.join('## '+name+'\n\n'+table(headers, rows) for name,headers,rows in sections)
    (OUT/'ALL_TABLES.md').write_text(report,encoding='utf-8')
    save('report_tables.json', sections)
    print(json.dumps(dict(validation=checks, attribution=aggregate, distribution=distribution),ensure_ascii=False))


if __name__ == '__main__':
    main()
