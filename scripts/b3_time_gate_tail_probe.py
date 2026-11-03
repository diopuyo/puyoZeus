"""時間門の最遅静止指標更新を1件だけ再現し、内訳を測る（門の値は置換しない）。"""
from __future__ import annotations

import json
from pathlib import Path
import time

from src import b3_features
from src.board import Board
from src.b3_source import iter_frames, resolve_attempt
from src.b3_timing import TimingTable
from scripts.b3_time_gate import ROOT, TABLE, Replay

MS = 1000.0


def main() -> None:
    output = ROOT / 'b3_data/time_gate'
    updates = [json.loads(line) for line in (output / 'realtime_30min/updates.jsonl').read_text().splitlines()]
    slowest = max(updates, key=lambda r: r['indicators_ms'])
    record = resolve_attempt(slowest['video'], ROOT / 'b3_eval_records').record
    table = TimingTable.from_json(TABLE.read_text())
    replay = Replay(record, slowest['video'], table)
    for frame in iter_frames(record, slowest['frame']):
        replay.step(frame, make_features=False)
    details = []
    for side in replay.gs.sides:
        if side.board is None:
            continue
        board = Board.from_list(side.board.astype(int).tolist())
        for name, func in b3_features._IND_FUNCS.items():
            start = time.perf_counter()
            func(board)
            details.append(dict(side=side.name, indicator=name, ms=(time.perf_counter()-start)*MS))
    result = dict(target=slowest, note='終了後のcold再現1件。正式分位点には混ぜない。',
                  details=sorted(details, key=lambda d: -d['ms']))
    (output / 'tail_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
