"""保存した起点入力を旧実装・融合直列・融合並列で照合し、単独時間を保存する。"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src import production_config, puyo_core_bridge as native
from src import exchange_prefire_candidates as reference
from src.b3_chain_prediction import origin_candidates

MS = 1000.0
PERCENTILES = (50, 95, 99, 100)


def run(cases: list[dict], mode: str) -> list[dict]:
    """完走結果の順序付き全要素を比較する。時間上限による打切りは行わない。"""
    adopted = production_config.B3_ORIGIN_NATIVE_ENABLED
    wrapper = native.b3_origin_candidates
    placements = reference.placements
    counts = [0, 0]

    def counted(*args: object) -> list:
        values = placements(*args)
        counts[0] += 1
        counts[1] += len(values)
        return values

    production_config.B3_ORIGIN_NATIVE_ENABLED = mode != 'reference'
    if mode == 'reference':
        reference.placements = counted
    if mode == 'serial':
        native.b3_origin_candidates = lambda b, p, h: wrapper(b, p, h, parallel=False)
    rows = []
    try:
        for case in cases:
            counts[:] = [0, 0]
            board = None if case['board'] is None else np.asarray(case['board'], dtype=np.int8)
            start = perf_counter()
            result = origin_candidates(board, tuple(case['colors']), tuple(case['queue']))
            elapsed = (perf_counter() - start) * MS
            digest = sha256(json.dumps(result, separators=(',', ':')).encode()).hexdigest()
            rows.append(dict(ms=elapsed, digest=digest, options=len(result[0]), reason=result[1],
                             placements_calls=counts[0], expanded=counts[1]))
    finally:
        production_config.B3_ORIGIN_NATIVE_ENABLED = adopted
        native.b3_origin_candidates = wrapper
        reference.placements = placements
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.inputs.read_text(encoding='utf-8-sig'))
    modes = {mode: run(cases, mode) for mode in ('reference', 'serial', 'parallel')}
    expected = [r['digest'] for r in modes['reference']]
    matches = all([r['digest'] for r in rows] == expected for rows in modes.values())
    summary = {mode: dict(zip(('p50', 'p95', 'p99', 'max'),
               np.percentile([r['ms'] for r in rows], PERCENTILES))) for mode, rows in modes.items()}
    result = dict(cases=len(cases), exact=matches, milliseconds=summary, runs=modes)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(cases=len(cases), exact=matches, milliseconds=summary)), flush=True)
    if not matches:
        raise AssertionError('起点探索の候補に差分があるため採用不可')


if __name__ == '__main__':
    main()
