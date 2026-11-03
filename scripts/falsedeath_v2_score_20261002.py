"""D-v2専用採点。旧原票を上書きせず、115試合門と旧門を並記する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import falsedeath_score_20261001 as prior
from scripts._diag_set2_falsedeath_audit import scan, read, events
from scripts.falsedeath_replay_20261001 import records

MODE = 'v2'
DEST = prior.OUT/'v2'
BOOT_N, BOOT_SEED, GAMES = 10_000, 0, 115
TIME_POINT_MAX, TIME_CI_MAX = .002, .020
AGREE_POINT_MIN, AGREE_CI_MIN = -.005, -.030
EVEN_THRESHOLD, REVERSAL_RATIO = 3., 1.2
CERTAIN = ('unavoidable_death','confirmed_death')


def visible_events(path: Path) -> list[dict]:
    """表示担当区間だけを返し、warmupの内部イベントと分離する。"""
    with np.load(path.parent/'display.npz') as data:
        begin,end = float(data['t_sec'][0]),float(data['t_sec'][-1])
    return [dict(e,values=[v for v in e['values'] if begin<=v['t_sec']<=end]) for e in events(path)]


def visible_scan(paths: list[Path], games: list[dict]) -> dict:
    """イベントの最初がwarmupでも、担当区間内に再出現した事例を落とさない。"""
    from scripts import _diag_set2_falsedeath_audit as audit
    original = audit.events
    try:
        audit.events = visible_events
        return scan(paths,games)
    finally:
        audit.events = original


def warmup_rows(mode: str) -> list[dict]:
    """18記録の表示外内部イベントを試合・側ごとに記録する。"""
    rows = []
    for source in records():
        directory = prior.OUT/'replay'/mode/source
        with np.load(directory/'display.npz') as data:
            begin,end = float(data['t_sec'][0]),float(data['t_sec'][-1])
        selected: dict[tuple,dict] = {}
        for event in events(directory/'events.jsonl'):
            for value in event['values']:
                if value['source'] not in CERTAIN or begin<=value['t_sec']<=end:
                    continue
                for side in value.get('dead_sides',()):
                    key = (event['game_idx'],side)
                    selected.setdefault(key,dict(record=source,game=key[0],side=side,
                        first_sec=value['t_sec'],source=value['source'],display_begin=begin,display_end=end))
        rows.extend(selected.values())
    return rows


def combined(sets: dict) -> dict:
    """試合をセット名で分離して結合し、固定seedの同じ添字で対差を取る。"""
    assert sum(v['games'] for v in sets.values())==GAMES
    index = np.random.default_rng(BOOT_SEED).integers(0,GAMES,(BOOT_N,GAMES))
    output = {}
    for metric in ('m_time','m_agree'):
        values = {mode:np.concatenate([prior.noise.per_game(s['rows'][mode],metric)
                  for s in sets.values()]) for mode in ('prod',MODE)}
        output[metric] = dict(paired=prior.noise.paired(values[MODE],values['prod'],index),
                             **{m:float(prior.noise.aggregate(v)) for m,v in values.items()})
    return dict(games=GAMES,bootstrap=dict(n=BOOT_N,seed=BOOT_SEED),metrics=output)


def metric_gates(value: dict) -> dict[str,bool]:
    """事前登録の不等号をそのまま使う。"""
    time,agree = (value['metrics'][k]['paired'] for k in ('m_time','m_agree'))
    return dict(m_time=time['diff']<=TIME_POINT_MAX and time['ci95'][1]<TIME_CI_MAX,
                m_agree=agree['diff']>=AGREE_POINT_MIN and agree['ci95'][0]>AGREE_CI_MIN)


def reversals(sets: dict) -> dict:
    """±3帯内は直前の方向を保ち、試合をまたぐ反転を数えない。"""
    result = {}
    for name,(parts,labels) in prior.SETS.items():
        games = read(labels)['games']
        counts = {}
        for mode in ('prod',MODE):
            data = prior.scorer.load_display([p/'display.npz' for p in prior.paths(mode,parts)])
            count = 0
            for game in games:
                mask = (data['t_sec']>=game['start'])&(data['t_sec']<game['end'])
                values = data['display_adv'][mask]
                signs = np.sign(values[np.abs(values)>EVEN_THRESHOLD])
                count += int(np.count_nonzero(np.diff(signs)))
            counts[mode] = count
        result[name] = dict(**counts,passed=counts[MODE]<=REVERSAL_RATIO*counts['prod'])
    return result


def old_gates(sets: dict, existing: dict, target: dict) -> dict:
    """旧条件を変更せず、warmup込みの旧監査値で別記する。"""
    return dict(false_death=existing['false']==existing['unresolved']==0 and all(
        s['deaths']['after']['false']==s['deaths']['after']['false_frames']==0 for s in sets.values()),
        m_time=all(s['metrics']['m_time']['paired']['ci95'][1]<prior.M_TIME_UPPER_MAX for s in sets.values()),
        m_agree=all(s['metrics']['m_agree']['paired']['ci95'][0]>prior.M_AGREE_LOWER_MIN for s in sets.values()),
        regression=target['passed'])


def evaluate_visible(terminals: dict) -> tuple[dict,dict]:
    """既存採点器の指標定義を保ち、イベントの監査区間だけ変更する。"""
    original_scan,original_events = prior.scan,prior.events
    try:
        prior.scan,prior.events = visible_scan,visible_events
        sets = {name:prior.score_set(name,MODE,terminals) for name in prior.SETS}
        existing = prior.existing(MODE,terminals)
    finally:
        prior.scan,prior.events = original_scan,original_events
    return sets,existing


def main(mode: str = 'v2') -> None:
    """梯子通過と18記録完了を先に確認し、新規採点だけを保存する。"""
    global MODE, DEST
    MODE, DEST = mode, prior.OUT/mode
    assert read(DEST/'LADDER.json')['passed'],'短区間の停止条件により全体採点禁止'
    prior.paths(MODE,list(records()))
    terminal = read(prior.OUT/'TERMINALS.json')['all']
    raw = {name:prior.score_set(name,MODE,terminal) for name in prior.SETS}
    original_existing,target = prior.existing(MODE,terminal),prior.regression(MODE)
    sets,existing = evaluate_visible(terminal)
    merged = combined(sets)
    flips = reversals(sets)
    gates = dict(**metric_gates(merged),false_death=existing['false']==existing['unresolved']==0 and all(
        s['deaths']['after']['false']==s['deaths']['after']['false_frames']==0 for s in sets.values()),
        regression=target['passed'],reversals=all(r['passed'] for r in flips.values()))
    legacy = old_gates(raw,original_existing,target)
    result = dict(mode=MODE,sets=sets,existing5=existing,combined=merged,regression=target,
        reversals=flips,gates_v2=gates,passed_v2=all(gates.values()),old_condition_gates=legacy,
        old_condition_passed=all(legacy.values()),old_condition_sets=raw,
        warmup={m:warmup_rows(m) for m in ('prod',MODE)})
    with (DEST/f'RESULT_{MODE}.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=1)
    print(json.dumps(dict(gates_v2=gates,old_condition_gates=legacy),ensure_ascii=False))


if __name__=='__main__':
    main()
