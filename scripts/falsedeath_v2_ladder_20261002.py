"""D-v2の段の梯子。4誤確定と3:00場面だけを先に検査する。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT, main as replay_main
from scripts._diag_set2_falsedeath_audit import events
from scripts.measure_e23_display_20260928 import displayed

DEST = OUT/'v2'
MODE = 'v2'
STOP = dict(c5=1736.2, mia8KCjr52g=870., review=2775.2)
CASES = (('c5',0,'2P',1728.1),('mia8KCjr52g',5,'1P',474.73333333333335),
         ('mia8KCjr52g',9,'2P',772.7666666666667),('mia8KCjr52g',10,'1P',847.8))
V2F_CASES = CASES + (('mia8KCjr52g',7,'2P',691.4333333333333),)
CERTAIN = ('unavoidable_death','confirmed_death')
SCENE_START, SCENE_END, DEADLINE, THRESHOLD = 2750.,2775.,2766.,.95
MIN_MEMORY_KIB = 6*1024**2


def save(name: str, data: Any) -> None:
    """新規出力だけを書き、再実行による上書きを拒否する。"""
    DEST.mkdir(exist_ok=True)
    with (DEST/name).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=1)


def run(source: str) -> None:
    """単一記録を元の開始から対象直後まで再生する。"""
    memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert os.getpriority(os.PRIO_PROCESS,0)==10
    assert int(memory['MemAvailable'].split()[0])>=MIN_MEMORY_KIB
    resource = dict(nice=10,memory_available_kib=int(memory['MemAvailable'].split()[0]))
    if os.environ.get('FALSEDEATH_CONTROLLER_MODULES'):
        resource['excluded_controller_modules'] = os.environ['FALSEDEATH_CONTROLLER_MODULES'].split()
    save(f'resources_{source}.json',resource)
    sys.argv = ['replay',MODE,source,'--out',str(DEST/'short'/source),'--end-sec',str(STOP[source])]
    replay_main()


def case_rows() -> list[dict]:
    """指定4事例の試合全体（c5はwarmup）で勝者側の死亡断定を検査する。"""
    rows = []
    for source,game,winner,stamp in V2F_CASES if MODE in ('v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b') else CASES:
        directory = DEST/'short'/source
        wrong = [v['t_sec'] for e in events(directory/'events.jsonl') if e['game_idx']==game
                 for v in e['values'] if v['source'] in CERTAIN and winner in v.get('dead_sides',())]
        with np.load(directory/'display.npz') as data:
            mask = (data['game_idx']==game)&np.isin(data['source'],CERTAIN)
            mask &= (data['display_p1']<.5 if winner=='1P' else data['display_p1']>.5)
            false_frames = int(mask.sum())
            index = int(np.argmin(abs(data['t_sec']-stamp))) if len(data['t_sec']) else None
            at = None if index is None or abs(float(data['t_sec'][index])-stamp)>.001 else dict(
                source=str(data['source'][index]),p1=float(data['display_p1'][index]))
        rows.append(dict(source=source,game=game,winner=winner,original_sec=stamp,
                         false_event_values=len(wrong),false_first_sec=min(wrong) if wrong else None,
                         false_frames=false_frames,at_original=at,warmup=source=='c5'))
    return rows


def scene() -> dict:
    """事前登録と同じ直接表示とEMA再適用の指標を使う。"""
    path = DEST/'short/review/display.npz'
    with np.load(path) as data:
        time,p = data['t_sec'],data['display_p1']
        hit = np.flatnonzero((time>=SCENE_START)&(time<=SCENE_END)&(p>=THRESHOLD))
        direct = float(time[hit[0]]) if len(hit) else None
        source = str(data['source'][hit[0]]) if len(hit) else None
    time,p = displayed(path)
    hit = time[(time>=SCENE_START)&(time<=SCENE_END)&(p>=THRESHOLD)]
    legacy = float(hit[0]) if len(hit) else None
    reasons = Counter(p['reason'] for e in events(path.parent/'events.jsonl') for v in e['values']
                      if 2760<=v['t_sec']<=2772 for p in v.get('post_counter_bound',()))
    return dict(first_display_sec=direct,legacy_metric_sec=legacy,source=source,
                deadline=DEADLINE,passed=legacy is not None and legacy<=DEADLINE,
                event_reason_counts=dict(reasons))


def score() -> None:
    """停止条件を機械判定し、全体再生の許可票を保存する。"""
    for source in STOP:
        assert (DEST/'short'/source/'status.json').exists(),source
    rows,target = case_rows(),scene()
    clean = all(r['false_event_values']==r['false_frames']==0 for r in rows)
    result = dict(cases=rows,cases_n=len(rows),scene=target,
                  gates=dict(false_death=clean,scene=target['passed']),passed=clean and target['passed'])
    save('LADDER.json',result)
    print(json.dumps(result,ensure_ascii=False))


def main() -> None:
    """再生と採点を分離し、run_limitedから起動する。"""
    global MODE, DEST
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=(*STOP,'score'))
    parser.add_argument('--mode',choices=('v2','v2b','v2c','v2d','v2e','v2f','v2g', 'v2h', 'v2h2', 'v2h2b'),default='v2')
    args = parser.parse_args()
    MODE, DEST = args.mode, OUT/args.mode
    score() if args.action=='score' else run(args.action)


if __name__=='__main__':
    main()
