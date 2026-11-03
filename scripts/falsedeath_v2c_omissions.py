"""E35″の未知ツモ省略を呼出し単位・同一試合の後続観測死亡で集計する。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT, records

DEST = OUT/'v2c'


def classify(rows: list[dict], terminals: list[dict], begin: float, end: float) -> list[dict]:
    """未観測は生存と断言せず、記録内に後続死亡が無いものとして数える。"""
    selected = []
    for row in rows:
        if not row.get('unknown_current_pair_omitted_self_chain', False):
            continue
        later = [r['stamp'] for r in terminals if r['game'] == row['game']
                 and r['side'] == row['side'] and r['stamp'] > row['t_sec']]
        selected.append(dict(game=row['game'], side=row['side'], chain_id=row['chain_id'],
            t_sec=row['t_sec'], dead=row['dead'], cached=row['cached'],
            in_display=begin <= row['t_sec'] <= end,
            later_observed_death=min(later) if later else None,
            observation=row['current_puyo_observation']))
    return selected


def counts(rows: list[dict]) -> dict:
    """キャッシュ再利用を含む実呼出し件数。表示外を別記する。"""
    observed = sum(r['later_observed_death'] is not None for r in rows)
    return dict(calls=len(rows), observed=observed, not_observed=len(rows)-observed,
                display_calls=sum(r['in_display'] for r in rows),
                outside_display_calls=sum(not r['in_display'] for r in rows),
                dead_calls=sum(r['dead'] for r in rows))


def main() -> None:
    """完成済み18記録の監査原票を集計し、判定条件へは追加しない。"""
    terminal = json.loads((OUT/'TERMINALS.json').read_text())['all']
    results = {}
    for source in records():
        directory = OUT/'replay/v2c'/source
        assert (directory/'status.json').exists()
        audit = json.loads((directory/'post_counter_bound_audit.json').read_text())['rows']
        with np.load(directory/'display.npz') as data:
            begin, end = float(data['t_sec'][0]), float(data['t_sec'][-1])
        rows = classify(audit, terminal[source], begin, end)
        results[source] = dict(**counts(rows), rows=rows)
    all_rows = [r for value in results.values() for r in value['rows']]
    result = dict(unit='E35証明の実呼出し（キャッシュ再利用を含む）',
        not_observed_definition='同一記録・同一試合・同一側の当該時刻より後に観測死亡なし。生存断定ではない。',
        total=counts(all_rows), records=results)
    with (DEST/'OMISSIONS.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    print(json.dumps(result['total'], ensure_ascii=False))


if __name__ == '__main__':
    main()
