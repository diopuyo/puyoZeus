"""報告用の状態・NEXT・通知・盤面の全変化点を整理する。"""
from __future__ import annotations

import json
from typing import Any

from scripts._diag_mia7_inspect import OUT, guard, save


def compact(board: list[list[int]]) -> str:
    """先頭の空行だけを省略し、全セルを復元できる表現にする。"""
    empty = 0
    while empty < len(board) and not any(board[empty]):
        empty += 1
    return f'空{empty}行;' + '/'.join(''.join(map(str, row)) for row in board[empty:])


def main() -> None:
    """原票の差分を内容別に保存する。"""
    guard()
    changes = json.loads((OUT / 'input_changes.json').read_text())
    previous: dict[str, Any] = {}
    events: dict[str, list] = {k: [] for k in ('states', 'next', 'chain', 'snapshot')}
    for row in changes:
        values = dict(states=[s['state'] for s in row['sides']],
            next=[(s['next'], s['dnext'], s['motion']) for s in row['sides']],
            chain=[s['chain'] for s in row['sides']], snapshot=row['snapshot'])
        for key, value in values.items():
            if previous.get(key) != value:
                events[key].append(dict(t=row['t'], value=value))
                previous[key] = value
    save('observed_changes.json', events)
    proof = json.loads((OUT / 'first_proof_inputs.json').read_text())
    chains = [{k: v for k, v in c.items() if 'board' not in k} for c in proof['chains'] if c['trigger_sec'] >= 618.9]
    print('CHAINS', json.dumps(chains, ensure_ascii=False))
    for key in ('chain', 'snapshot'):
        print(key, json.dumps(events[key], ensure_ascii=False))
    reconcile = json.loads((OUT / 'placement_signal_reconcile.json').read_text())
    print('NEXT_RECONCILE', json.dumps([{k: r[k] for k in ('t_sec','side','reason','signal')} for r in reconcile]))
    print('BOARDS')
    for row in json.loads((OUT / 'stable_boards.json').read_text()):
        print(round(row['t'], 3), row['side'], compact(row['board']))


if __name__ == '__main__':
    main()
