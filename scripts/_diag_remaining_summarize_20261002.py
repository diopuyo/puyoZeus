"""抽出済み原票の膨大な探索枝を省き、比較可能な観測値を表示する。"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from scripts._diag_remaining_20261002 import OUT, read, save

KEYS = ('t_sec', 'source', 'p1', 'incoming', 'death_incoming', 'pending_ledger',
        'state_safety', 'hands', 'dead_sides', 'held_from_sec')
CHAIN_KEYS = ('side', 'chain_id', 'trigger_sec', 'score_before', 'score_delta', 'formula_total',
              'display_score', 'score_ready_sec', 'score_ready_reason', 'end_signal_sec', 'post_end_drop_sec')


def short(value: dict) -> dict:
    """判定経路の理由のみを抜き出す。"""
    result = {k: value[k] for k in KEYS if k in value}
    for key in ('single_death_proof', 'post_counter_bound', 'multi_landing'):
        result[key] = [{k: v for k, v in x.items() if k in ('side', 'dead', 'reason', 'undecided_reason', 'nodes')}
                       if x else None for x in value.get(key, [])]
    return result


def main() -> None:
    """各モードの対象時点と理由遷移を比較する。"""
    output: dict[str, Any] = {}
    for case in read(OUT / 'false_cases.json'):
        tag, stamp = f"{case['source']}_g{case['game']}", case['first_sec']
        output[tag] = {}
        for mode in ('all', 'prod', 'no_d3'):
            data = read(OUT / f'{tag}_{mode}_events.json')
            values = [v for e in data['events'] for v in e['values']]
            exact = [short(v) for v in values if abs(v['t_sec']-stamp) < .001]
            previous = [short(v) for v in values if v['t_sec'] <= stamp][-3:]
            chains = [{k: c[k] for k in CHAIN_KEYS} for e in data['events'] for c in e['chains']]
            output[tag][mode] = dict(display=data['nearest_display'], exact=exact, previous=previous, chains=chains)
    for mode in ('all', 'no_d2', 'prod'):
        data = read(OUT / f'scene_{mode}_events.json')
        rows = [dict(t_sec=v['t_sec'], **p) for e in data['events'] for v in e['values']
                if 2760 <= v['t_sec'] <= 2772 for p in v.get('post_counter_bound', [])]
        counts = Counter(r['reason'] for r in rows)
        output[f'scene_{mode}'] = dict(n=len(rows), counts=dict(counts),
            ranges={k: [min(r['t_sec'] for r in rows if r['reason']==k),
                        max(r['t_sec'] for r in rows if r['reason']==k)] for k in counts})
    save('compact_comparison.json', output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
