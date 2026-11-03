"""E35のコミット済み実装を独立ロードし、同一入力で速度と全証明を比較する。"""
from __future__ import annotations

import argparse
import cProfile
import gzip
import importlib.util
import json
from pathlib import Path
import pstats
import subprocess
import sys
from time import perf_counter
from types import ModuleType

import numpy as np

from src.board import Board
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e35b')
REFERENCE = '0f28a0495c4ba2b360b49324d880da43ab316dce'


def reference() -> ModuleType:
    """共有worktreeの並行変更に依存しない旧上限計算を読む。"""
    path = OUT/'reference.py'
    git_dir = Path('.git').read_text().strip().removeprefix('gitdir: ')
    if sys.platform == 'linux' and ':/' in git_dir:
        git_dir = f'/mnt/{git_dir[0].lower()}/{git_dir[3:]}'
    path.write_bytes(subprocess.check_output(['git', f'--git-dir={git_dir}', 'show',
                                             f'{REFERENCE}:src/post_counter_death_bound.py']))
    spec = importlib.util.spec_from_file_location('e35b_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quantiles(seconds: list[float]) -> dict:
    """実計算のP50/P95をミリ秒で保存する。"""
    return dict(n=len(seconds), p50_ms=float(np.percentile(seconds, 50)*1000),
                p95_ms=float(np.percentile(seconds, 95)*1000), total_sec=sum(seconds))


def inputs(source_filter: str | None = None) -> list[dict]:
    """元の5記録順に、キャッシュヒットを含む全入力を復元する。"""
    from scripts.run_e35 import SOURCES
    rows = []
    for source in SOURCES:
        if source_filter and source != source_filter:
            continue
        with gzip.open(OUT/f'{source}.inputs.json.gz', 'rt') as stream:
            rows.extend(dict(source=source, **r) for r in json.load(stream))
    return rows


def calculate(module: ModuleType, row: dict) -> list[dict]:
    """候補順とNEXT・おじゃま・時間を保持して旧APIで証明する。"""
    audit = row['audit']
    return [module.prove_post_counter(Board.from_list(board), tuple(audit['queue']),
            audit['incoming'], audit['hands'], row['elapsed'], tuple(audit['palette']))
            for board in row['boards']]


def timed_group(module: ModuleType, row: dict, engine: object | None) -> tuple[list[dict], float]:
    """盤面復号を時間外に置き、本番と同じ計算境界を測る。"""
    audit = row['audit']
    boards = [Board.from_list(b) for b in row['boards']]
    args = (tuple(audit['queue']), audit['incoming'], audit['hands'], row['elapsed'], tuple(audit['palette']))
    started = perf_counter()
    result = (engine.proofs_for(boards, *args) if engine else
              [module.prove_post_counter(board, *args) for board in boards])
    return result, perf_counter()-started


def profile(module: ModuleType, rows: list[dict], label: str = 'baseline') -> None:
    """段別の累積時間と自己時間を同じ原票から得る。"""
    profiler = cProfile.Profile()
    profiler.enable()
    for row in rows:
        if not row['audit']['cached']:
            assert calculate(module, row) == row['audit']['proofs']
    profiler.disable()
    profiler.dump_stats(str(OUT/f'{label}.prof'))
    with (OUT/f'PROFILE_{label}.txt').open('w') as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('tottime').print_stats(25)


def paired(module: ModuleType, rows: list[dict]) -> None:
    """前後を入力ごとに交互の順で測り、全証明辞書を照合する。"""
    import src.post_counter_death_bound as current
    from src.exchange_post_counter_bound import PostCounterDeathBound
    from src.post_counter_geometry import geometry, component, has_four, reply_bytes
    geometry.cache_clear()
    component.cache_clear()
    has_four.cache_clear()
    reply_bytes.cache_clear()
    module.score_upper.cache_clear()
    module.step_upper.cache_clear()
    current.score_upper.cache_clear()
    current.step_upper.cache_clear()
    timings: dict[str, list[float]] = dict(before=[], after=[])
    proofs, game, engine = 0, None, None
    memo: dict = {}
    for index, row in enumerate(rows):
        audit = row['audit']
        boundary = (row['source'], audit['game'])
        if game != boundary:
            game, engine, memo = boundary, PostCounterDeathBound(), {}
        key = (tuple(map(str, row['boards'])), audit['incoming'], tuple(audit['queue']), audit['hands'],
               tuple(audit['palette']), current.compute_effective_rate(row['elapsed']))
        assert audit['cached'] == (key in memo)
        if key in memo:
            assert memo[key] == audit['proofs']
            continue
        order = [('before', module), ('after', current)]
        for name, implementation in (order if index % 2 else reversed(order)):
            result, duration = timed_group(implementation, row, engine if name == 'after' else None)
            timings[name].append(duration)
            assert result == audit['proofs'], (name, index, row['source'], result, audit)
        memo[key] = result
        proofs += len(result)
    save_json(OUT/'BENCHMARK.json', dict(before=quantiles(timings['before']), after=quantiles(timings['after']),
              matched_rows=len(rows), matched_proofs=proofs, reference=REFERENCE,
              cache_info={f.__name__:f.cache_info()._asdict() for f in (geometry, component, has_four, reply_bytes)}))


def main() -> None:
    """採取前の小標本または全5記録を段別測定する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--profile-current', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    module = reference()
    if args.sample:
        sample = json.loads(Path('logs/e35/D4_SCENE_BOUND.json').read_text())['proofs']
        rows = [dict(boards=[r['board']], elapsed=0, audit=dict(cached=False, queue=[4,5,1,5],
                incoming=171, hands=1, palette=[1,3,4,5], proofs=[r['proof']])) for r in sample]
    else:
        rows = inputs(args.source)
    if args.compare:
        paired(module, rows)
    elif args.profile_current:
        import src.post_counter_death_bound as current
        profile(current, rows, 'current')
    else:
        profile(module, rows)


if __name__ == '__main__':
    main()
