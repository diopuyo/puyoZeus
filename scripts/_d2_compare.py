"""D2の短区間再収集と保存原票を時刻・盤面・状態で厳密比較する。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from scripts._d2_inventory import OUT, save

FIELDS = ('state', 'confirmed_board', 'score', 'next_pair', 'dnext_pair')


def lines(path: Path) -> list[dict]:
    """完了済みgzip原票だけを読み出す。"""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def recorded(path: Path, times: set[float]) -> list[dict]:
    """対象時刻の更新行を元の精度のまま取得する。"""
    result = []
    last = max(times)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'update':
                continue
            args = row['args']['tuple']
            if args[3] > last:
                break
            if args[3] not in times:
                continue
            sides = args[0]['namespace']
            result.append(dict(t_sec=args[3], sides=[sides[k]['namespace'] for k in ('p1', 'p2')]))
    return result


def compare(left: list[dict], right: list[dict], fields: tuple[str, ...]) -> dict:
    """欠測や時刻ずれも不一致として数える。"""
    assert len(left) == len(right), (len(left), len(right))
    counters = {key: 0 for key in fields}
    different, first = 0, None
    for before, after in zip(left, right):
        assert before['t_sec'] == after['t_sec']
        keys = [key for key in fields if any(a.get(key) != b.get(key)
                for a, b in zip(before['sides'], after['sides']))]
        for key in keys:
            counters[key] += 1
        if keys:
            different += 1
            if first is None:
                first = dict(t_sec=before['t_sec'], keys=keys, before=before, after=after)
    return dict(rows=len(left), different_rows=different, per_field=counters, first=first)


def all_runs() -> tuple[dict, dict]:
    """完了条件ごとに旧・新の保存盤面との一致数を出す。"""
    traces, results = {}, {}
    for marker in sorted(OUT.glob('*/complete.json')):
        root = marker.parent
        metadata = json.loads((root/'command.json').read_text())
        source = metadata['source']
        trace = lines(root/'trace.jsonl.gz')
        traces[root.name] = trace
        times = {r['t_sec'] for r in trace}
        results[root.name] = dict(source=source, variant=metadata['variant'])
        for stage in ('e31', 'e34b'):
            reference = recorded(Path(f'logs/{stage}/records/{source}.jsonl.gz'), times)
            results[root.name][stage] = compare(reference, trace, FIELDS)
    return traces, results


def main() -> None:
    """全条件の一致数と最初の認識中間値の差を保存する。"""
    traces, results = all_runs()
    pairs = [('q_old', 'q_new'), ('q_new', 'q_new2'), ('q_sig', 'q_sig2'),
             ('q_old', 'q_sig'), ('q_old', 'q_vote'), ('q_new2', 'q_sig2')]
    pairs += [(f'{s}_sig', f'{s}_new') for s in ('fc', 'mia', 'zenchi', 'review')]
    comparisons = {}
    for left, right in pairs:
        if left in traces and right in traces:
            fields = tuple(k for k in traces[left][0]['sides'][0]
                           if k in traces[right][0]['sides'][0])
            comparisons[f'{left}:{right}'] = compare(traces[left], traces[right], fields)
    configs = {p.parent.name: json.loads(p.read_text()) for p in OUT.glob('*/config.json')}
    differences = {}
    if 'q_old' in configs and 'q_new' in configs:
        old, new = configs['q_old'], configs['q_new']
        differences = {k: dict(old=old[k], new=new[k]) for k in old if old[k] != new[k]}
    save('comparisons.json', dict(runs=results, pairs=comparisons, effective_differences=differences))
    print(json.dumps(dict(runs={k: {s: v[s]['per_field']['confirmed_board'] for s in ('e31', 'e34b')}
                               for k, v in results.items()}, config_diff=differences), indent=2))


if __name__ == '__main__':
    main()
