"""重力待ち末尾を一フレームでも観測できたか、状態遷移の原票を確認する。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np

from src.exchange_event_record import read_records
from src.exchange_midchain_completion import compact, remaining
from src.exchange_event_landing import ExchangeLandingProjection
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e26')


def main() -> None:
    """評価値を使わず、末尾観測の支持・UNKNOWN・残り連鎖だけを監査する。"""
    previous, formulas, counts, rows = [None, None], [None, None], Counter(), []
    simulator = ExchangeLandingProjection().simulator
    for item in read_records(OUT/'records/review.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, *_ = item['args']
        for idx, side in enumerate((result.p1, result.p2)):
            before = previous[idx]
            if before and before[1].state.name == 'GRAVITY_SETTLE' and side.state.name == 'CHAIN':
                board, formula = before[1].midchain_board, formulas[idx]
                valid = board is not None and compact(board)
                value = remaining(board, *formula, simulator) if valid and formula else None
                reason = 'remaining_chain' if value else ('no_chain' if valid else 'invalid_board')
                counts[reason] += 1
                rows.append(dict(t_sec=stamp, sample_sec=before[0], side=idx+1, reason=reason,
                    formula=formula, unknown=int(np.count_nonzero(board._grid == 10)) if board else None,
                    next_formula=None if side.chain_event is None else vars(side.chain_event),
                    prediction=value))
                rows[-1]['next_formula'] = {k: v for k, v in (rows[-1]['next_formula'] or {}).items()
                                           if k != 'before_board'}
            event = side.chain_event
            if event and event.mechanism == 'formula_read':
                formulas[idx] = (event.chain_count, event.total_score)
            previous[idx] = (stamp, side)
    save_json(OUT/'SETTLE_DIAGNOSTIC.json', dict(counts=dict(counts), transitions=rows))
    print(dict(counts), flush=True)
    print([r for r in rows if r['t_sec'] >= 2751. and r['side'] == 2], flush=True)


if __name__ == '__main__':
    main()
