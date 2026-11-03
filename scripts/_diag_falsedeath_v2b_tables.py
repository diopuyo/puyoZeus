"""D-v2b短区間の全表を、既存原票を変更せず抽出する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT
from scripts._diag_set2_falsedeath_audit import events
from scripts.measure_e23_display_20260928 import displayed

DEST = OUT/'v2b'
CERTAIN = ('unavoidable_death','confirmed_death')
SCENE_START, SCENE_END, THRESHOLD = 2750.,2775.,.95
GIB_KIB = 1024**2
TIME_TOLERANCE = .001


def read(path: Path) -> dict:
    """UTF-8の原票だけを読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def enrich(case: dict) -> dict:
    """対象試合の表示母数と、warmupの同時刻内部出力を添える。"""
    directory = DEST/'short'/case['source']
    with np.load(directory/'display.npz') as data:
        count = int(np.count_nonzero(data['game_idx']==case['game']))
    at = [v for e in events(directory/'events.jsonl') if e['game_idx']==case['game']
          for v in e['values'] if abs(v['t_sec']-case['original_sec'])<TIME_TOLERANCE]
    internal = {k:at[0].get(k) for k in ('t_sec','source','p1','dead_sides')} if at else None
    return dict(case,display_frames_n=count,internal_at_original=internal)


def warmup(source: str) -> list[dict]:
    """表示開始前の内部死亡イベントを、試合・側・経路で集約する。"""
    with np.load(OUT/'replay/prod'/source/'display.npz') as data:
        begin = float(data['t_sec'][0])
    rows: dict[tuple,dict] = {}
    for event in events(DEST/'short'/source/'events.jsonl'):
        for value in event['values']:
            if value['t_sec']>=begin or value['source'] not in CERTAIN:
                continue
            for side in value['dead_sides']:
                key = (event['game_idx'],side,value['source'])
                row = rows.setdefault(key,dict(record=source,game=key[0],side=side,source=key[2],
                    first_sec=value['t_sec'],last_sec=value['t_sec'],values_n=0,display_begin=begin))
                row['last_sec'] = value['t_sec']
                row['values_n'] += 1
    return list(rows.values())


def scene(path: Path) -> dict:
    """同じ1場面で、実表示と従来のEMA再適用を併記する。"""
    with np.load(path) as data:
        time,prob = data['t_sec'],data['display_p1']
        hit = np.flatnonzero((time>=SCENE_START)&(time<=SCENE_END)&(prob>=THRESHOLD))
        direct = float(time[hit[0]]) if len(hit) else None
        source = str(data['source'][hit[0]]) if len(hit) else None
    time,prob = displayed(path)
    hit = time[(time>=SCENE_START)&(time<=SCENE_END)&(prob>=THRESHOLD)]
    return dict(direct_sec=direct,legacy_sec=float(hit[0]) if len(hit) else None,source=source)


def main() -> None:
    """集計JSONは新規作成し、再生・採点原票には書かない。"""
    ladder = read(DEST/'LADDER.json')
    sources = ('c5','mia8KCjr52g','review')
    statuses = {s:read(DEST/'short'/s/'status.json') for s in sources}
    resources = {s:read(DEST/f'resources_{s}.json') for s in sources}
    for resource in resources.values():
        resource['memory_available_gib'] = resource['memory_available_kib']/GIB_KIB
    directories = {m:OUT/'replay'/m/'review' for m in ('prod','all','no_d2')}
    directories.update({m:OUT/m/'short/review' for m in ('v2','v2b')})
    result = dict(cases=[enrich(c) for c in ladder['cases']],statuses=statuses,resources=resources,
        scenes={m:scene(p/'display.npz') for m,p in directories.items()},
        warmup={s:warmup(s) for s in sources},full_replay_exists=(OUT/'replay/v2b').exists(),
        total_replay_seconds=sum(s['elapsed_seconds'] for s in statuses.values()))
    assert not ladder['passed'] and not result['full_replay_exists']
    with (DEST/'TABLES.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=1)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
