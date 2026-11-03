"""保存済み指定場面だけで、複数着弾探索の診断を出力する。"""
from __future__ import annotations

import json
from pathlib import Path
import time

from src.board import Board
from src.exchange_event_landing import ExchangeLandingProjection
from src.exchange_event_multilanding import prove_multilanding
from scripts.run_e3_exchange_eval_20260926 import save_json


def main() -> None:
    """確実性が成立する最初の盤面から、全着弾の探索を点検する。"""
    rows = json.loads(Path('logs/e23/E22_SCENE_TRACE.json').read_text())
    row = next(r for r in rows if r['sides'][1]['certain'])
    side = row['sides'][1]
    start = time.perf_counter()
    result = prove_multilanding(Board.from_list(side['response_board']), tuple(side['queue']),
        side['incoming'], side['hands'], row['elapsed'], ExchangeLandingProjection()._optimistic_response)
    save_json(Path('logs/e23/SCENE_PROBE.json'), dict(t_sec=row['t_sec'],
        elapsed_seconds=time.perf_counter()-start, **result))
    print({k: v for k, v in result.items() if k != 'rounds'}, flush=True)


if __name__ == '__main__':
    main()
