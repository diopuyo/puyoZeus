"""隠し段候補が使えなかった理由を、入力記録だけから分類する。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np
from src.board_state_machine import BoardState
from src.exchange_event_record import read_records
from src.match_color_evidence import MatchColorEvidence
from src.exchange_midchain_completion import MAX_SETTLE_GAP_SEC
from src.exchange_event_tracker import TIME_EPSILON_SEC
from scripts.run_e3_exchange_eval_20260926 import save_json


def diagnose(source: str) -> dict:
    """未来フレームを使わずに色集合と落ち際の適格条件を追跡する。"""
    game, evidence, previous, rows, counts = None, [], [None, None], [], Counter()
    for item in read_records(Path('logs/e26/records')/f'{source}.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, idx, *_ = item['args']
        if idx != game:
            game, evidence = idx, [MatchColorEvidence(), MatchColorEvidence()]
            previous = [None, None]
        for tracker, side in zip(evidence, (result.p1, result.p2)):
            tracker.observe(side)
        for idx, side in enumerate((result.p1, result.p2)):
            board = getattr(side, 'midchain_board', None)
            if side.state != BoardState.GRAVITY_SETTLE or board is None:
                previous[idx] = None
                continue
            grid = board._grid
            colors = evidence[idx].active()
            unknown = int(np.count_nonzero(grid[0] == 10))
            reasons = dict(hidden_outside_limit=not 1 <= unknown <= 3,
                visible_unknown=bool(np.any(grid[1:] == 10)),
                floating=bool(np.any((grid[:-1] != 0) & (grid[1:] == 0))),
                colors_unconfirmed=len(colors) != 4)
            counts.update(k for k, v in reasons.items() if v)
            old, identity = previous[idx], (grid.tobytes(), colors)
            repeat = (old is not None and old[0] == identity
                and TIME_EPSILON_SEC < stamp-old[1] <= MAX_SETTLE_GAP_SEC+TIME_EPSILON_SEC)
            samples = old[2]+1 if repeat and not any(reasons.values()) else int(not any(reasons.values()))
            previous[idx] = (identity, stamp, samples)
            if 2750 <= stamp <= 2766:
                rows.append(dict(t_sec=stamp, game=game, side=idx+1, unknown=unknown,
                    colors=sorted(colors), settle_samples=samples, reasons=reasons))
    return dict(source=source, overlapping_reasons=dict(counts), target_rows=rows)


def main() -> None:
    """指定場面の除外条件を保存する。"""
    value = diagnose('review')
    save_json(Path('logs/e27/HIDDEN_DIAGNOSTIC.json'), value)
    print(value['overlapping_reasons'])


if __name__ == '__main__':
    main()
