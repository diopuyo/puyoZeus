"""5Bの同一100局面照合と、既知全深さの通知コストを別々に測る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from scripts import prefire_v5_experiment as reference
from src import prefire_v5_search as base
from src import prefire_v5_value as value
from src import prefire_v5b_search as exact
from src.prefire_v5b_value import CachedM0

OUT = Path('logs/prefire_prediction/v5b')


def proxy(overlay: SimpleNamespace) -> SimpleNamespace:
    """モデルは同じ個体を使用し、計算済み側encoderだけ再利用する。"""
    return SimpleNamespace(**{**vars(overlay), '_m0': CachedM0(overlay._m0)})


def check(row: dict, overlay: SimpleNamespace) -> dict:
    """独立Python参照に対しnative全列挙・厳密枝刈り・encoder再利用を照合する。"""
    original = value.choose
    optimized = proxy(overlay)
    def choose(states: tuple, attacker: int, elapsed: float, evaluator: value.ValueFunction,
               depth: int = 1, k: int | None = None) -> tuple | None:
        return exact.choose(states, attacker, elapsed,
                            lambda exchange, time: value.evaluate(optimized, exchange, time), depth)
    value.choose = choose
    try:
        return reference.check_sample(row, overlay)
    finally:
        value.choose = original


def benchmark(row: dict, overlay: SimpleNamespace) -> dict:
    """候補列挙だけでなく、両側局面価値・相殺・着弾を含む1通知を測る。"""
    states = tuple(base.Position(bytes(g), tuple(k), p)
                   for g, k, p in zip(row['boards'], row['known'], row['pending']))
    from src.prefire_best_play_v5 import BestPlayV5Layer
    layer = BestPlayV5Layer()
    start = perf_counter()
    layer._schedule_v5(overlay, (), states, (base.OK, base.OK), row['elapsed'], row['t_sec'])
    best = layer._cache[()]
    return dict(source=row['source'], t_sec=row['t_sec'], milliseconds=(perf_counter()-start)*1000,
                candidates=[len(exact.candidates(p, side=i)) for i, p in enumerate(states)],
                best=[b[0] for b in best])


def main() -> None:
    """既存出力を上書きせず再開できる。採点と速度計測の母数は分ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=100)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    samples = json.loads((reference.OUT/'samples_ledger.json').read_text())
    dest = OUT/('timing' if args.benchmark else 'verification')
    dest.mkdir(parents=True, exist_ok=True)
    overlay = reference.model_overlay()
    for index in range(args.start, args.stop):
        path = dest/f'case_{index:03d}.json'
        if path.exists():
            continue
        result = (benchmark if args.benchmark else check)(samples[index], overlay)
        path.write_text(json.dumps(result), encoding='utf-8')
        print(index, result, flush=True)


if __name__ == '__main__':
    main()
