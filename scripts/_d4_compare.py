"""同値検収済み観測再生のcount・隠し段・モデル入力を対比較する。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scripts._d4_loss import OUT, Q, save
from scripts._d3_inventory import decode
from scripts.r1_measure_helpers import lines

MODES = ('off', 'on')
CHUNK = 10000


def bounds() -> dict:
    """実学習行の列別最小最大を一定メモリで算出する。"""
    result = {}
    for model in ('S1', 'S3'):
        data = np.load(f'logs/e15/{model}_prime.npy', mmap_mode='r')
        low, high = np.full(data.shape[1], np.inf), np.full(data.shape[1], -np.inf)
        for start in range(0, len(data), CHUNK):
            low = np.minimum(low, np.nanmin(data[start:start+CHUNK], axis=0))
            high = np.maximum(high, np.nanmax(data[start:start+CHUNK], axis=0))
        result[model] = dict(shape=list(data.shape), low=low.tolist(), high=high.tolist())
    return result


def outside(call: dict, support: dict) -> dict:
    """軸ごとの範囲逸脱を測る。分布外の十分条件とはしない。"""
    model = call['name']
    if model not in support:
        return {}
    x = np.array(call['features'])
    low, high = np.array(support[model]['low']), np.array(support[model]['high'])
    if len(x) != len(low):
        return dict(dimension_mismatch=[len(x), len(low)])
    idx = np.flatnonzero((x<low) | (x>high))
    return dict(n=len(idx), columns=idx.tolist())


def compare(source: str, support: dict) -> dict:
    """共通時刻で欠測も含め比較し、最大悪化点の全入力を保存する。"""
    cohort = json.loads((OUT/'q_rows.json').read_text()) if source==Q else []
    stamps = {round(r['t'], 6) for r in cohort}
    top = json.loads((OUT/'q_top10_seconds.json').read_text()) if source==Q else []
    peaks = {round(r['peak']['t'], 6) for r in top}
    if source=='review':
        peaks = {round(r['t'], 6) for r in json.loads((OUT/'scene_inputs.json').read_text())}
    streams = [lines(OUT/'replay'/m/source/'trace.jsonl.gz') for m in MODES]
    counts, examples, differences = Counter(), [], []
    last = {m: {} for m in MODES}
    for a, b in zip(*streams):
        assert a['t']==b['t'] and a['game']==b['game']
        pair = dict(zip(MODES, [decode(a), decode(b)]))
        for mode, row in pair.items():
            for call in row['model_calls']:
                last[mode][call['name']] = call
        valid = round(a['t'], 6) in stamps if source==Q else 2740<=a['t']<=2772
        if not valid:
            continue
        counts['n'] += 1
        flags = {k: pair['off'][k] != pair['on'][k] for k in ('counts','hidden','chains','exchange','source')}
        identities = [[tuple(c[k] for k in ('side','chain_id','trigger_sec','observed_sec'))
            for c in pair[m]['chains']] for m in MODES]
        flags['event_identity'] = identities[0] != identities[1]
        counts.update(k for k,v in flags.items() if v)
        for mode, row in pair.items():
            for call in row['model_calls']:
                support_row = outside(call, support)
                if support_row:
                    counts[mode+'_model_calls'] += 1
                    counts[mode+'_outside_calls'] += bool(support_row.get('n'))
        differences.append(dict(t=a['t'], **flags))
        if round(a['t'],6) in peaks or source=='review' and abs(a['t']-2759.05)<.01:
            examples.append(dict(t=a['t'], **pair, last_models={m: dict(last[m]) for m in MODES}))
    save(f'{source}_trace_summary.json', dict(counts))
    save(f'{source}_trace_examples.json', examples)
    save(f'{source}_trace_difference.json', differences)
    return dict(counts)


def main() -> None:
    """OFF/ONの保存出力が一致検収済みのときだけ解析する。"""
    support = bounds()
    save('training_support.json', support)
    for source in (Q, 'review'):
        if all((OUT/'replay'/m/source/'EQUIVALENCE.json').exists() for m in MODES):
            print(source, compare(source, support), flush=True)


if __name__ == '__main__':
    main()
