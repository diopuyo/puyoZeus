"""出力一致の確認: 発火前の側特徴 (prefire_side_features) を native 連鎖計算で求めても Python 版と同じ値か。

入力は本番構成の5記録 (exev logs/pending_expiry/full/records) の STABLE 確定盤面と記録の NEXT/NEXT2。
重複を除いた盤面を記録ごとに上限 SAMPLE_LIMIT 件 (先頭から順に) 取り、経過秒は ELAPSED_LEVELS を順に当てる
(特徴が経過秒に依存するのは換算率だけ)。1件でも不一致なら native を使わない。
使い方: PYTHONPATH=. python -m scripts.prefire_native_feature_parity_20261001 --source q_7gc4TgFig
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.board import Board
from src.prefire_best_play import feature_simulator
from src.exchange_event_features import prefire_side_features
from src.exchange_event_record import read_records
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, RECORDS

SAMPLE_LIMIT = 1500
ELAPSED_LEVELS = (30.0, 120.0, 200.0, 300.0)   # マージンタイム前後の換算率を含む
OUT = Path('logs/prefire_prediction/native_parity')


def samples(source: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """STABLE 確定盤面と NEXT/NEXT2 (未読は 0) の重複なし標本。"""
    seen, out = set(), []
    for row in read_records(RECORDS / f'{source}.jsonl.gz'):
        if row['kind'] != 'update':
            continue
        for side in (row['args'][0].p1, row['args'][0].p2):
            board = side.confirmed_board
            if getattr(getattr(side, 'state', None), 'value', None) != 'stable' or board is None:
                continue
            grid = (board if isinstance(board, Board) else Board.from_list(np.asarray(board).tolist()))._grid
            queue = np.array([*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))])
            key = grid.tobytes() + queue.tobytes()
            if key not in seen:
                seen.add(key)
                out.append((grid.copy(), queue))
            if len(out) >= SAMPLE_LIMIT:
                return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=tuple(BASELINE_DIRS), required=True)
    source = parser.parse_args().source
    mismatched, errors, fallback = [], 0, 0
    items = samples(source)
    for i, (grid, queue) in enumerate(items):
        elapsed = ELAPSED_LEVELS[i % len(ELAPSED_LEVELS)]
        try:
            left = prefire_side_features(grid, queue, elapsed)
        except ValueError:
            errors += 1
            continue   # Python 版が入力を拒む盤面は比較の母数から外す (件数は残す)
        simulator = feature_simulator(grid)   # 浮きぷよ・UNKNOWN の盤面は Python 版へ戻す (本番の振り分けと同じ)
        fallback += simulator is None
        right = prefire_side_features(grid, queue, elapsed, simulator=simulator)
        if not np.array_equal(left, right):
            mismatched.append(dict(index=i, python=left.tolist(), native=right.tolist()))
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(source=source, samples=len(items), rejected_inputs=errors,
                  compared=len(items) - errors, python_fallback=fallback, mismatched=len(mismatched),
                  examples=mismatched[:5])
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}), flush=True)


if __name__ == '__main__':
    main()
