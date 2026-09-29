"""同じ実記録で、同期保持と次の正しい手番への追従を検証する。"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from src.board_state_machine import BoardState
from src.exchange_event_record import read_records
from src.exchange_event_sync import CountTurnSynchronizer

RECORD = Path("logs/review_zenchi_g41_43_e14/inputs.jsonl.gz")
OUT = Path("logs/e16/sync_trace.csv")
START, END = 2611.9, 2614.1


def main() -> None:
    """推論と同じ同期器を先頭から進め、採用盤面とNEXTの時刻を保存する。"""
    game, sync, rows = None, [], []
    for item in read_records(RECORD):
        if item["kind"] != "update":
            continue
        result, _, _, stamp, current_game, *_ = item["args"]
        if current_game != game:
            game, sync = current_game, [CountTurnSynchronizer(), CountTurnSynchronizer()]
        for i, side in enumerate((result.p1, result.p2)):
            grid = side.confirmed_board._grid if side.confirmed_board is not None else None
            queue = np.array([*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))])
            accepted = sync[i].observe(grid, queue, stamp, side.state == BoardState.STABLE,
                side.next_slide_motion, side.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE))
            saved = sync[i].accepted
            if START <= stamp <= END:
                rows.append(dict(t_sec=stamp, side=i+1, updated=accepted, reason=sync[i].reason,
                    accepted_sec=saved.t_sec if saved else None,
                    accepted_puyos=int(np.count_nonzero(saved.grid)) if saved else None,
                    accepted_queue=saved.queue.tolist() if saved else None,
                    current_pair=sync[i].current_pair, candidates=len(sync[i].candidates)))
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [r for r in rows if r["side"] == 1 and r["updated"]]
    print(selected, flush=True)


if __name__ == "__main__":
    main()
