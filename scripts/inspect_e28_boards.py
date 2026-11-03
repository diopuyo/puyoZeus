"""映像確認用の落ち際盤面と独立シミュレーションを保存する。"""
from pathlib import Path
import numpy as np
from src.exchange_event_record import read_records
from src.exchange_midchain_completion import remaining
from src.chain import ChainSimulator
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from scripts.run_e3_exchange_eval_20260926 import save_json


def main() -> None:
    """採用条件を緩和せず、診断だけに一枚の盤面を使用する。"""
    rows, formula = [], [None, None]
    simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for item in read_records(Path('logs/e26/records/review.jsonl.gz')):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, *_ = item['args']
        for idx, side in enumerate((result.p1, result.p2)):
            event = side.chain_event
            # 候補盤面はそのフレーム以前に読めた式だけと組み合わせる。
            before = formula[idx]
            if event is not None and event.mechanism == 'formula_read':
                formula[idx] = (event.chain_count, event.total_score)
            board = getattr(side, 'midchain_board', None)
            if not 2754 <= stamp <= 2771 or board is None:
                continue
            value = remaining(board, *before, simulator) if before else None
            rows.append(dict(t_sec=stamp, side=idx+1, formula_before=before,
                formula_after=formula[idx], board=board._grid.tolist(),
                unknown=int(np.count_nonzero(board._grid == 10)),
                floating=bool(np.any((board._grid[:-1] != 0) & (board._grid[1:] == 0))),
                simulation=value))
    save_json(Path('logs/e28/BOARD_DIAGNOSTIC.json'), rows)
    print([(r['t_sec'], r['side'], r['formula_before'], r['simulation']['score'])
           for r in rows if r['simulation']], flush=True)


if __name__ == '__main__':
    main()
