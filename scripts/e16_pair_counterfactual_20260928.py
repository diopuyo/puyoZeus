"""盤面とNEXTを交差させ、欠測・手番ずれの影響を分離する。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from src.exchange_event_record import read_records
from src.exchange_event_count_features import fire

OUT = Path("logs/e16")
TIMES = (2613.2166666666667, 2613.516666666667, 2613.616666666667, 2613.8166666666666)
INCOMING_EXACT = 388.7142857142857


def main() -> None:
    """全時刻で換算率70を固定し、同じ1手応手計算で比較する。"""
    chosen = {}
    for item in read_records(Path("logs/review_zenchi_g41_43_e14/inputs.jsonl.gz")):
        if item["kind"] == "update" and item["args"][3] in TIMES:
            side = item["args"][0].p1
            queue = (*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0)))
            chosen[item["args"][3]] = (side.confirmed_board._grid.astype(np.int8), queue)
    results = []
    for board_time, (grid, _) in chosen.items():
        for next_time, (_, queue) in chosen.items():
            valid = tuple(v if 1 <= v <= 5 else 0 for v in queue)
            available = float(fire(grid.tobytes(), valid, 0., (-1,), True)[0])
            results.append(dict(board_sec=board_time, next_sec=next_time, next=queue,
                available=available, margin=available-INCOMING_EXACT))
    before, after = chosen[TIMES[0]][0], chosen[TIMES[-1]][0]
    delta = [dict(row=int(r), col=int(c), before=int(before[r, c]), after=int(after[r, c]))
             for r, c in np.argwhere(before != after)]
    output = dict(crossed=results, board_changes=delta)
    (OUT / "pair_counterfactual.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
