"""保存rawに次手正常対照を置けるかを小さく確認する。"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_belief_hidden_landing_2026-09-11_v1'))
import hidden_landing as H


def main() -> None:
    data = json.loads((ROOT.parent / 'video38_history_publication_probe_live_2026-09-11_v12/LIVE_EMPTY_RESET.json').read_bytes())
    raw = [r['raw'] for r in data['recovery'] if r['kind'] == 'reset_baseline_wait'][-1]
    B = H.B
    grid = B.grid(B.Board.from_dict({'grid': raw}))
    rows = []
    for h in H.enumerate_hypotheses(grid, (4, 5)):
        if h.touches_hidden:
            rows.append(dict(cells=h.cells, hidden=True))
            continue
        result = B.ChainSimulator(exclude_hidden_row_from_pop=True).simulate(
            B.Board.from_dict({'grid': H.apply_hypothesis(grid, h)}))
        rows.append(dict(cells=h.cells, hidden=h.touches_hidden,
                         chain=result.chain_count, death=result.final_board.get(1, 2)))
    print(json.dumps(dict(raw=raw, candidates=rows)))


if __name__ == '__main__':
    main()
