"""保存盤面の変化とシミュレーション結果を照合する診断。"""
from __future__ import annotations
from collections import Counter
import json
import gzip
from pathlib import Path
import numpy as np
from src.board import Board
from src.chain import ChainSimulator
from scripts._d1_inventory import OUT, SOURCES

HIDDEN_ROWS, RECENT_SECONDS = 1, 10


def probe(source: str, rows: list[dict]) -> list[dict]:
    """前後の盤面を保存し、同一シミュ結果になる変化だけを記す。"""
    inventory = json.loads((OUT/f'{source}_inventory.json').read_text())
    with gzip.open(f'logs/e31/records/{source}.jsonl.windows.json.gz','rt') as stream:
        windows = json.load(stream)
    simulator, results = ChainSimulator(), []
    for row in rows:
        if not row['cells']:
            continue
        fire = next(f for f in inventory['fires'] if f['game'] == row['game'] and int(f['side']=='p2') == row['side'] and f['chain_event']['trigger_sec'] == row['trigger'])
        window = next(w for w in windows if w['game']==row['game'] and w['side']==row['side'] and w['trigger_sec']==row['trigger'])
        simulated = simulator.simulate(Board.from_list(window['board']))
        sim_matches = np.array_equal(simulated.final_board._grid[HIDDEN_ROWS:],np.array(fire['origin'])[HIDDEN_ROWS:])
        changes = [x for x in inventory['changes'] if x['confirmed_board'] is not None and x['game'] == row['game'] and int(x['side']=='p2') == row['side'] and x['t'] < row['trigger']]
        transitions = []
        for before, after in zip(changes, changes[1:]):
            if after['t'] < row['trigger']-RECENT_SECONDS:
                continue
            a, b = np.array(before['confirmed_board']), np.array(after['confirmed_board'])
            result = simulator.simulate(Board.from_list(a.tolist()))
            matches = np.array_equal(result.final_board._grid[HIDDEN_ROWS:], b[HIDDEN_ROWS:])
            transitions.append(dict(t=after['t'], previous_t=before['t'], state=after['state']['state'],
                count_before=int(np.count_nonzero(a[HIDDEN_ROWS:])), count_after=int(np.count_nonzero(b[HIDDEN_ROWS:])),
                delta=[[int(r),int(c),int(a[r,c]),int(b[r,c])] for r,c in np.argwhere(a!=b)],
                exact_simulation=matches, chains=result.chain_count, before=a.tolist(), after=b.tolist()))
        results.append(dict(source=source, game=row['game'], side=row['side'], trigger=row['trigger'],
            origin=fire['origin'], snapshot=window['board'], snapshot_simulation_matches=sim_matches,
            snapshot_simulation=simulated.final_board._grid.tolist(), snapshot_chains=simulated.chain_count,
            cells=row['cells'], transitions=transitions))
    return results


def main() -> None:
    """照合根拠を後の分類と画像読取りに渡す。"""
    differences = json.loads((OUT/'differences.json').read_text())
    results = [v for s in SOURCES for v in probe(s, [r for r in differences if r['source']==s])]
    (OUT/'transitions.json').write_text(json.dumps(results, indent=2))
    for row in results:
        if len(row['cells']) > 4:
            print(row['source'], row['trigger'], 'snapshot_sim',row['snapshot_simulation_matches'],row['snapshot_chains'], [(round(t['t'],3),t['count_before'],t['count_after'],t['exact_simulation'],t['chains']) for t in row['transitions'] if abs(t['count_after']-t['count_before'])>2])


if __name__ == '__main__':
    main()
