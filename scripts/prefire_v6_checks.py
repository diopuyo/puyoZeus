"""Phase 6採点前の基準再生一致と軽量盤面評価の単体速度を測る。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.prefire_v6_replay import OUT, EVALSET
from scripts.prefire_v6_measure import save
from src.prefire_v6_value import board_value


def off_controls() -> dict:
    """保存済み本番と差があった区間を、同じworktreeのOFF全長再生で照合する。"""
    result = {}
    for part in ('s1', 's5'):
        with np.load(OUT/'baseline'/part/'display.npz') as left, np.load(OUT/'replay'/part/'display.npz') as right:
            equal = {column:bool(np.array_equal(left[column], right[column]))
                     for column in ('t_sec', 'display_p1', 'display_adv', 'source')}
            result[part] = dict(rows=len(left['t_sec']), equal=equal, passed=all(equal.values()))
    if not all(row['passed'] for row in result.values()):
        raise AssertionError('補正0のON/OFFが不一致。通常評価との差の切分け未完了')
    return result


def main() -> None:
    """セット2短窓を本番s1の同時刻と全表示列で照合する。"""
    with np.load(EVALSET/'set2/replay/prod/s1/display.npz') as left, np.load(OUT/'baseline/check_v2/display.npz') as right:
        stamps = right['t_sec']
        indices = np.searchsorted(left['t_sec'], stamps)
        columns = ('t_sec', 'display_p1', 'display_adv', 'source')
        equal = {c:bool(np.array_equal(left[c][indices], right[c])) for c in columns}
        errors = {c:float(np.max(np.abs(left[c][indices]-right[c]))) for c in columns[:-1]}
    rows = json.loads(Path('logs/prefire_prediction/v5_experiment/samples_ledger.json').read_text())
    costs = []
    for row in rows:
        for raw in row['boards']:
            board_value.cache_clear()
            start = perf_counter()
            board_value(bytes(raw))
            costs.append((perf_counter()-start)*1000)
    result = dict(baseline_rows=len(stamps), equal=equal, max_error=errors,
                  light_board_ms=dict(count=len(costs), p95=float(np.percentile(costs, 95)), maximum=max(costs)),
                  passed=all(equal.values()))
    save(OUT/'CHECKS_v2.json', result)
    print(result, flush=True)
    if not result['passed']:
        raise AssertionError('本番基準再生が不一致のため採点禁止')


if __name__ == '__main__':
    main()
