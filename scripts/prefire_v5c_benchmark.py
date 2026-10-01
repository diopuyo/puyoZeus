"""5Cの通知全体をcold/warmで実測し、採点前のLの根拠を保存する。"""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
from pathlib import Path
from time import perf_counter

from scripts import prefire_v5_experiment as reference
from src import prefire_v5_search as base
from src import prefire_v5b_search as exact
from src.prefire_best_play_v5 import BestPlayV5Layer
from src.prefire_v5c_search import side_options

OUT = Path('logs/prefire_prediction/v5c/notifications')


def measure(row: dict, profile: bool = False) -> dict:
    """ロード時間と区別し、列挙・応手・両側評価を含む通知を測る。"""
    overlay = reference.model_overlay()
    layer = BestPlayV5Layer()
    states = tuple(base.Position(bytes(g), tuple(k), p)
                   for g, k, p in zip(row['boards'], row['known'], row['pending']))
    side_options.cache_clear()
    exact.candidates.cache_clear()
    profiler = cProfile.Profile()
    if profile:
        profiler.enable()
    start = perf_counter()
    layer._schedule_v5(overlay, ('cold',), states, (base.OK, base.OK), row['elapsed'], row['t_sec'])
    milliseconds = (perf_counter()-start)*1000
    if profile:
        profiler.disable()
        profiler.dump_stats(str(OUT/'after.prof'))
        with (OUT/'after.txt').open('w') as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
    cold = layer._cache[('cold',)]
    start = perf_counter()
    layer._schedule_v5(overlay, ('warm',), states, (base.OK, base.OK), row['elapsed'], row['t_sec'])
    warm_ms = (perf_counter()-start)*1000
    return dict(source=row['source'], t_sec=row['t_sec'], milliseconds=milliseconds,
        warm_ms=warm_ms, warm_equal=layer._cache[('warm',)] == cold, profile=profile,
        candidates=[len(exact.candidates(p, side=i)) for i, p in enumerate(states)],
        best=[r[0] for r in cold], evaluations=layer._evaluation.misses,
        evaluation_hits=layer._evaluation.hits,
        landing_cache=layer._transitions.land.cache_info()._asdict())


def main() -> None:
    """既定の5記録代表通知を個別プロセスで計測する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=int)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    name = ('profile' if args.profile else 'benchmark') + f'_{args.index:03d}.json'
    path = OUT/name
    if path.exists():
        raise FileExistsError(path)
    samples = json.loads((reference.OUT/'samples_ledger.json').read_text())
    result = measure(samples[args.index], args.profile)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
