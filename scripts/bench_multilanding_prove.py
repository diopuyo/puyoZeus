"""保存した prove_multilanding の入力 (profile_*.jsonl の inputs) を単体で再実行し、所要と一致を比べる (段2)。

映像・認識・tracker を経由せず、証明関数だけを実データの入力で走らせるので、高速化の反復が数秒〜分で回る。
- 基準の結果は記録の result_sha (json 全文の SHA-256) と reason/nodes/dead。一致しなければ不一致として数える。
- 各呼出しの前に future_send の lru を捨てる (単独の費用。本番の連続呼出しでは命中で安くなる)。
使い方: python -m scripts.bench_multilanding_prove logs/multilanding_speed/base_head/profile_*.jsonl [--min-sec 0.1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.board import Board
from src.exchange_event_landing import NEAR_FUTURE_KNOWN_HAND_SLOTS, future_send
from src.exchange_event_multilanding import prove_multilanding

SUMMARY_KEYS = ('reason', 'nodes', 'dead')


def optimistic(board: Board, queue: np.ndarray, hands: int, elapsed: float) -> float:
    """`ExchangeLandingProjection._optimistic_response` と同じ呼び出し。"""
    grid = board._grid
    return future_send(grid.tobytes(), grid.shape, grid.dtype.str,
                       tuple(int(v) for v in queue), hands + NEAR_FUTURE_KNOWN_HAND_SLOTS, elapsed)


def load_calls(paths: list[Path], min_sec: float) -> list[dict]:
    """入力付きの prove 行だけを集める (出典名を付ける)。"""
    calls = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['fn'] == 'prove_multilanding' and 'inputs' in row and row['sec'] >= min_sec:
                calls.append(dict(row, source=path.stem.removeprefix('profile_')))
    return calls


def run_call(row: dict) -> tuple[dict, float]:
    """1 呼出しを再実行して (結果, 秒) を返す。"""
    spec = row['inputs']
    board = Board()
    board._grid = np.asarray(spec['grid'], dtype=board._grid.dtype)
    future_send.cache_clear()
    started = perf_counter()
    result = prove_multilanding(board, tuple(spec['queue']), spec['incoming_in'], spec['hands_in'],
                                spec['elapsed_in'], optimistic, spec['credit_in'])
    return result, perf_counter() - started


def same(row: dict, result: dict) -> bool:
    """記録した結論・ノード数・全文 SHA (あれば) の一致。"""
    ok = all(row[k] == result.get(k) for k in SUMMARY_KEYS)
    if 'result_sha' in row:
        digest = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
        ok = ok and digest == row['result_sha']
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--min-sec', type=float, default=0.0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    calls = load_calls(args.paths, args.min_sec)
    calls = calls[:args.limit] if args.limit else calls
    rows: list[dict[str, Any]] = []
    for row in calls:
        result, sec = run_call(row)
        rows.append(dict(source=row['source'], old_sec=row['sec'], new_sec=sec, nodes=row['nodes'],
                         reason=row['reason'], match=same(row, result), sha_checked='result_sha' in row))
    mismatches = [r for r in rows if not r['match']]
    print(json.dumps(dict(calls=len(rows), mismatches=len(mismatches),
                          sha_checked=sum(r['sha_checked'] for r in rows),
                          old_total=sum(r['old_sec'] for r in rows), new_total=sum(r['new_sec'] for r in rows),
                          new_max=max((r['new_sec'] for r in rows), default=0.0))), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
