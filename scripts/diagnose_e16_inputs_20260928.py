"""E16の変更前に盤面・NEXTの時系列を損失なく保存する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from src.exchange_event_record import read_records
from src.board import BOARD_ROWS, BOARD_COLS

OUT = Path("logs/e16")
RECORD = Path("logs/review_zenchi_g41_43_e14/inputs.jsonl.gz")
WINDOWS = ((2611.9, 2614.1), (2698.5, 2701.1))
PROTOCOL = dict(baseline="3385544", production_enabled=False,
    prerequisite="counter_margin反転が盤面/NEXT同期ずれの場合だけ実装修正。他原因なら停止報告",
    replay=dict(q_log_loss_max=.5518, zenchi_agreement_min=.7955,
                false_fire_max=1, false_fire_denominator=28),
    scenes=dict(prefire_window=[2612., 2614.08], prefire_mean="E15より高い",
                counter_sign_flips=0, terminal_window=[2699.5, 2701.], terminal_p1_min=.6),
    cv=dict(required="項目1で特徴が変わる場合のみ", same_rows=84445, folds=15, auc_min=.702))


def main() -> None:
    """保存済み入力の更新を追い、認識器を変更せず診断する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "PROTOCOL.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == PROTOCOL
    protocol.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, boards, previous = [], [], [None, None]
    for item in read_records(RECORD):
        if item["kind"] != "update":
            continue
        result, _, _, stamp, _, _, displayed, _ = item["args"]
        for idx, side in enumerate((result.p1, result.p2)):
            grid = side.confirmed_board._grid if side.confirmed_board is not None else None
            key = None if grid is None else grid.tobytes()
            queue = (side.next_pair, side.dnext_pair)
            changed = previous[idx] != (key, queue)
            previous[idx] = (key, queue)
            if not any(start <= stamp <= end for start, end in WINDOWS):
                continue
            chain = side.chain_event
            rows.append(dict(t_sec=stamp, side=idx+1, state=side.state.name, changed=changed,
                board_index=len(boards), puyos=None if grid is None else int(np.count_nonzero(grid)),
                next=side.next_pair, dnext=side.dnext_pair, slide=side.next_slide_motion,
                score=displayed[idx], chain_trigger=getattr(chain, "trigger_sec", None),
                chain_count=getattr(chain, "chain_count", None), mechanism=getattr(chain, "mechanism", None)))
            boards.append(np.zeros((BOARD_ROWS, BOARD_COLS), np.int8) if grid is None else grid)
    with (OUT / "input_timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(OUT / "input_boards.npy", np.asarray(boards))


if __name__ == "__main__":
    main()
