"""既存の固定100局面で案Cの冷キャッシュ1通知を測る。"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.plan_c_input_audit import memory_available, OUT
from src import prefire_threat_features as features
from src import prefire_v5_search as base
from src import prefire_v5b_search as exact
from src.prefire_v5c_search import side_options

SAMPLES = Path('logs/prefire_prediction/v5_experiment/samples_ledger.json')
P95_LIMIT_MS = 20.


def main() -> None:
    """両側の列挙・応手・変換を含め、欠測通知を分離する。"""
    rows = []
    engine = base.native._native
    optimized = engine is not None and hasattr(engine, 'prefire_threat_maxima_py')
    for index, sample in enumerate(json.loads(SAMPLES.read_text())):
        memory_available()
        positions = tuple(base.Position(bytes(grid), tuple(queue))
                          for grid, queue in zip(sample['boards'], sample['known']))
        base.status.cache_clear()
        exact.candidates.cache_clear()
        side_options.cache_clear()
        start = perf_counter()
        result = features.threat_features(positions, sample['elapsed'], enabled=True)
        values = features.encode(result)
        ms = (perf_counter() - start)*1000
        # 最適化前の5C応手表でも同じ特徴になるか、速度計測の外で照合する。
        reference = tuple(side_options(p, features.KNOWN_HANDS, side) for side, p in enumerate(positions))
        table = features.TransitionTable()
        expected = tuple(features._side(reference, side, sample['elapsed'], table) for side in range(2))
        np.testing.assert_array_equal(features.encode(expected), values)
        rows.append(dict(index=index, ms=ms, missing_sides=sum(t.attack_missing for t in result),
                         finite_columns=int(np.isfinite(values).sum()), parity=True,
                         raw=[t.raw().tolist() for t in result]))
    complete = [row['ms'] for row in rows if row['missing_sides'] == 0]
    report = dict(samples=len(rows), complete_samples=len(complete),
                  p95_ms=float(np.percentile([row['ms'] for row in rows], 95)),
                  complete_p95_ms=float(np.percentile(complete, 95)) if complete else None,
                  competing_jobs=True, quality_gate_measured=False, optimized_native=optimized, rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    name = 'BENCHMARK_MAXIMA.json' if optimized else 'BENCHMARK_REFERENCE.json'
    (OUT/name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
