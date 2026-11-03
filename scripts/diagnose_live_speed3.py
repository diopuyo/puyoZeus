"""保存計時を重複計上せず集計する。品質合格判定には使用しない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path('C:/puyo1_tools/soak')
OUT = Path(_public_dev_text('<work>/speed3'))
PERCENTILES = (50, 95, 99, 100)
MS = 1000


def rows(path: Path) -> Iterator[dict]:
    """途中停止した保存列は完全な行だけ読む。"""
    if path.suffix == '.pickle':
        with path.open('rb') as stream:
            while True:
                try:
                    yield pickle.load(stream)
                except EOFError:
                    return
    else:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)


def stats(values: list[float]) -> dict:
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=float(np.mean(values)),
                p=np.percentile(values, PERCENTILES).tolist())


def group(row: dict) -> str:
    states = (row['state1'], row['state2'])
    return 'chain' if 'CHAIN' in states else 'ojama' if 'OJAMA_FALL' in states else 'other'


def event_costs(path: Path) -> tuple[dict, dict]:
    costs, totals = {}, defaultdict(float)
    if not path.exists():
        return costs, dict(totals)
    for row in rows(path):
        stages = {k: v['wall_ms'] for k, v in row['stages'].items()}
        costs[row['frame']] = costs.get(row['frame'], 0) + sum(
            stages.get(k, 0) for k in ('event_update', 'event_reference'))
        for key, value in stages.items():
            totals[key] += value
    return costs, dict(totals)


def analyze(name: str) -> dict:
    root = ROOT / ('run_'+name)
    spool = root / 'output/spool'
    evaluations = {r['frame']: r for r in rows(spool/'evaluations.pickle')}
    child, totals = event_costs(spool/'frame_timing_event_evaluation.jsonl')
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows(spool/'frame_timing_evaluation.jsonl'):
        saved = evaluations.get(row['frame'])
        if saved is None:
            continue
        recognized = row['details']['recognized_at']
        latency = (row['finished_at']-row['captured_at'])*MS
        parts = dict(recognition=(recognized-row['captured_at'])*MS,
                     queue=(row['started_at']-recognized)*MS,
                     **(dict(child_update_reference=child[row['frame']],
                             parent_ipc_other=row['wall_ms']-child[row['frame']])
                        if row['frame'] in child else dict(evaluation_including_ipc=row['wall_ms'])))
        for label in ('all', group(saved)):
            for key, value in dict(parts, latency=latency, service=row['wall_ms'],
                                   depth=saved['queue_depth']).items():
                groups[label][key].append(value)
    result = {label: {key: stats(v) for key, v in vals.items()} for label, vals in groups.items()}
    threshold = result['all']['latency']['p'][1]
    for label, vals in groups.items():
        result[label]['above_all_p95'] = sum(v >= threshold for v in vals['latency'])
        result[label]['share_pct'] = {key: sum(vals[key])/sum(vals['latency'])*100 for key in parts}
    polls = [r['lat_ms'] for r in rows(root/'polls.jsonl') if r.get('shown') is not None]
    result['shown_polls'] = stats(polls)
    result['child_stage_total_ms'] = totals
    result['unmatched_evaluations'] = len(evaluations)-result['all']['latency']['n']
    return result


def main() -> None:
    OUT.mkdir(exist_ok=True)
    result = dict(status='diagnostic_not_quality_pass', runs={n: analyze(n) for n in ('g', 'h')},
                  note='探索の内側は未計装。child_update_referenceを探索占有率と呼ばない。IPC残差は親側他処理も含む。')
    (OUT/'soak_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
