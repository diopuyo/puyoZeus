"""E35⁗′の抑止を、同一記録・試合・側の後続観測死亡だけで集計する。"""
from __future__ import annotations

import json
import gzip
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT, records
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h2b'


def classify(rows: list[dict], terminals: list[dict], begin: float, end: float) -> list[dict]:
    """抑止呼出しと連鎖単位を区別し、未観測を生存の証拠にはしない。"""
    result = []
    for row in rows:
        check = row.get('completion_prefix_check')
        if check is None or check['accepted']:
            continue
        later = [v['stamp'] for v in terminals if v['game'] == row['game']
                 and v['side'] == row['side'] and v['stamp'] > row['t_sec']]
        result.append(dict(game=row['game'], side=row['side'], chain_id=row['chain_id'],
            t_sec=row['t_sec'], in_display=begin <= row['t_sec'] <= end,
            later_observed_death=min(later) if later else None, check=check))
    return result


def counts(rows: list[dict]) -> dict:
    """監査へ記録した抑止呼出し数と、記録内の独立連鎖数を返す。"""
    seen = sum(r['later_observed_death'] is not None for r in rows)
    return dict(calls=len(rows), later_death_observed=seen, later_death_not_observed=len(rows)-seen,
        chains=len({(r.get('record'), r['game'], r['side'], r['chain_id']) for r in rows}),
        display_calls=sum(r['in_display'] for r in rows),
        reasons=dict(Counter(r['check']['reason'] for r in rows)))


def past_counts(audit: list[dict], begin: float, end: float) -> dict:
    """過去段不一致は抑止だけでなく、許可した呼出しも含めて数える。"""
    rows = [r for r in audit if r.get('completion_prefix_check', {}).get('earlier_mismatch')]
    return dict(calls=len(rows), accepted=sum(r['completion_prefix_check']['accepted'] for r in rows),
        blocked=sum(not r['completion_prefix_check']['accepted'] for r in rows),
        proven_dead=sum(r.get('dead', False) for r in rows),
        display_calls=sum(begin <= r['t_sec'] <= end for r in rows),
        chains=len({(r['game'], r['side'], r['chain_id']) for r in rows}))


def summarize(full: bool) -> dict:
    """梯子失敗なら短区間だけを集計し、全体の数値を作らない。"""
    terminals = json.loads((OUT/'TERMINALS.json').read_text())['all']
    result = {}
    for source in records() if full else ('c5', 'mia8KCjr52g', 'review'):
        directory = OUT/'replay/v2h2b'/source if full else DEST/'short'/source
        audit = json.loads((directory/'post_counter_bound_audit.json').read_text())['rows']
        with np.load(directory/'display.npz') as data:
            begin = float(data['t_sec'][0]) if len(data['t_sec']) else float('inf')
            end = float(data['t_sec'][-1]) if len(data['t_sec']) else -float('inf')
        rows = classify(audit, terminals[source], begin, end)
        for row in rows:
            row['record'] = source
        result[source] = dict(**counts(rows), rows=rows, past=past_counts(audit, begin, end))
    rows = [r for group in result.values() for r in group['rows']]
    total = dict(**counts(rows), past={k: sum(r['past'][k] for r in result.values())
                                      for k in next(iter(result.values()))['past']})
    value = dict(scope='full' if full else 'short', total=total, records=result,
        definition='E35探索前の抑止呼出し数。後続死亡は同一試合・同一側で集計。未観測は生存断定ではない。')
    with gzip.open(DEST/'BLOCKS.json.gz', 'xt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    return value


def render(value: dict) -> str:
    """全記録の抑止数・後続観測の内訳を省略しない。"""
    text = table('E35⁗′による抑止（'+value['scope']+'）',
        ['記録', '抑止呼出し', '後続死亡あり', '後続死亡なし', '連鎖数', '表示区間', '理由'],
        [[name, *[r[k] for k in ('calls', 'later_death_observed', 'later_death_not_observed',
                                'chains', 'display_calls', 'reasons')]]
         for name, r in dict(value['records'], total=value['total']).items()]) + value['definition']+'\n\n'
    return text + table('過去段不一致を記録した件数',
        ['記録', '呼出し', '照合許可', '照合抑止', '死亡証明', '表示区間', '連鎖数'],
        [[name, *[r['past'][k] for k in ('calls', 'accepted', 'blocked', 'proven_dead', 'display_calls', 'chains')]]
         for name, r in dict(value['records'], total=value['total']).items()])
