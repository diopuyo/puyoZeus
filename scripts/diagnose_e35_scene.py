"""D4の保存済み正しい候補盤面でE35上限を診断する（採点入力には混ぜない）。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from src.board import Board
from src.chain import ChainSimulator
from src.exchange_prefire_candidates import completion
from src.hidden_row_belief import combinations
from src.probabilistic_board import ProbabilisticCell
from src.post_counter_death_bound import prove_post_counter
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e35')
STAMP = 2759.05


def prove_candidate(value: dict, queue: tuple) -> dict:
    """候補ごとの所要時間と証明を一緒に保存する。"""
    started = perf_counter()
    result = prove_post_counter(Board.from_list(value['board']), queue, 171, 1, 0, (1,3,4,5))
    return dict(board=value['board'], score=value['score'], proof=result,
                elapsed_sec=perf_counter()-started)


def main() -> None:
    """当該時点以前の式だけで候補を絞り、全候補を個別に証明する。"""
    audit = json.loads(Path('logs/d4/replay/on/review/prefire_audit.json').read_text())
    row = next(r for r in audit['rows'] if r['side']=='2P' and 2750 <= r['trigger_sec'] <= 2755)
    cells = [ProbabilisticCell({int(k):v for k,v in cell.items()}) for cell in row['hidden_distributions']]
    choices, _ = combinations(cells)
    simulator = ChainSimulator(exclude_hidden_row_from_pop=True)
    options = []
    for hidden, weight in choices:
        board = Board.from_list(row['snapshot']['board'])
        board._grid[0] = hidden
        candidate = completion(board, simulator)
        if all(v['count'] <= len(candidate['prefix']) and candidate['prefix'][v['count']-1] == v['score']
               for v in row['observations'] if v['t_sec'] <= STAMP):
            options.append(dict(candidate, weight=weight))
    maximum = max(v['score'] for v in options)
    boards = {tuple(map(tuple,v['board'])):v for v in options}
    scene = next(v for v in json.loads(Path('logs/d4/scene_inputs.json').read_text()) if v['t']==STAMP)
    side = scene['on']['input'][0]['p2']
    queue = tuple(side['next_pair'] + side['dnext_pair'])
    proofs = [prove_candidate(v, queue) for v in boards.values()]
    result = dict(scope='D4保存候補の診断のみ。本番固定入力と独立', t_sec=STAMP,
                  maximum_score=maximum, candidates=len(options), boards=len(boards),
                  all_dead=all(r['proof']['dead'] for r in proofs), proofs=proofs)
    save_json(OUT/'D4_SCENE_BOUND.json', result)
    print({k:v for k,v in result.items() if k!='proofs'}, flush=True)


if __name__ == '__main__':
    main()
