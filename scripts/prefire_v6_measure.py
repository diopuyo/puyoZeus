"""Phase 6の固定100局面診断と、他ジョブなしの初回通知測定。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from scripts import prefire_v5_experiment as reference
from scripts import prefire_v5b_replay as replay
from src import prefire_v5_search as base
from src import prefire_v5b_search as exact
from src import prefire_v6_value as light
from src.prefire_v5c_search import TransitionTable, side_options
from src.prefire_v5c_value import EvaluationCache
from src.prefire_best_play_v6 import BestPlayV6Layer, Strength
from src.prefire_v6_production import ProductionValue
from src.ojama_accounting import ON_FIELD_CAP

OUT = Path('logs/prefire_prediction/v6')
POLL_SEC = 5


class MeasurementBusy(RuntimeError):
    """他ジョブとの重複を正式値に混ぜず、同じ通知を後で再測定する。"""


def save(path: Path, result: dict) -> None:
    """原票を新規保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


def states(row: dict) -> tuple:
    """5Bの固定入力をそのまま復元する。"""
    return tuple(base.Position(bytes(g), tuple(k), p)
                 for g, k, p in zip(row['boards'], row['known'], row['pending']))


def snapshot(overlay: object, row: dict) -> None:
    """5Bで固定済みの両側予告から、S3に必要な同時刻会計列を復元する。"""
    left, right = row['pending']
    overlay._snapshots = [(row['t_sec'], SimpleNamespace(pending_p1=left, pending_p2=right,
        pending_p1_uncapped=left, pending_p2_uncapped=right, net_ojama_balance=right-left,
        net_balance_capped=min(right, ON_FIELD_CAP)-min(left, ON_FIELD_CAP),
        forecast_p1=left, forecast_p2=right))]


def diagnose(index: int, row: dict, overlay: object) -> dict:
    """5Bと同じ両側1手の独立診断。全3組通知の速度測定とは分ける。"""
    positions, table = states(row), TransitionTable()
    evaluator = ProductionValue(overlay, positions, row['elapsed'])
    result = []
    for side in (0, 1):
        picked = light.select(positions, side, row['elapsed'], table, depth=reference.REFERENCE_DEPTH)
        full = exact.choose(positions, side, row['elapsed'], evaluator, depth=reference.REFERENCE_DEPTH,
                            resolver=table.resolve, options_fn=side_options)
        # 選択手の本番価値は本番評価での最悪応手を取り直して測る。
        def options(p: base.Position, depth: int | None, which: int) -> tuple:
            return (picked[1],) if which == side else side_options(p, depth, which)
        selected = exact.choose(positions, side, row['elapsed'], evaluator,
                                depth=reference.REFERENCE_DEPTH, resolver=table.resolve, options_fn=options)
        result.append(dict(side=side, same_move=picked[1].path == full[1].path,
            regret_logit=(1 if side == 0 else -1)*(light.logit(full[0])-light.logit(selected[0])),
            light_path=picked[1].path, production_path=full[1].path,
            production_best=full[0], selected_production=selected[0]))
    return dict(index=index, source=row['source'], t_sec=row['t_sec'], depth=reference.REFERENCE_DEPTH,
                sides=result)


def idle_snapshot() -> dict:
    """pgrepで他Pythonを記録し、常駐OS更新待ち以外があれば測定しない。"""
    output = subprocess.run(['pgrep', '-af', 'python'], capture_output=True, text=True, check=False)
    lines = output.stdout.splitlines()
    others = [line for line in lines if int(line.split()[0]) != os.getpid()
              and Path(line.split()[1]).name.startswith('python')
              and '/usr/share/unattended-upgrades/' not in line]
    available = next(int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:'))
    return dict(time=time.time(), pgrep=lines, other_python=others, available_kib=available)


def benchmark(index: int, row: dict, overlay: object) -> dict:
    """全列挙・遷移表・両側の本番値を含む、空キャッシュの1通知。"""
    before = idle_snapshot()
    if before['other_python'] or before['available_kib'] < 4*1024*1024:
        raise MeasurementBusy('他Pythonまたはavailable不足のため正式測定を開始しない')
    layer = BestPlayV6Layer(strength=Strength((0.,)*5, identity=True))
    side_options.cache_clear()
    exact.candidates.cache_clear()
    light.board_value.cache_clear()
    start = time.perf_counter()
    layer._schedule_v5(overlay, ('cold',), states(row), (base.OK, base.OK), row['elapsed'], row['t_sec'])
    milliseconds = (time.perf_counter()-start)*1000
    after = idle_snapshot()
    if after['other_python']:
        save(OUT/'rejected'/f'{index:03d}_{time.time_ns()}.json',
             dict(before=before, after=after, milliseconds=milliseconds, valid=False))
        raise MeasurementBusy('測定中に他Pythonが出現したため正式値を採用しない')
    return dict(index=index, milliseconds=milliseconds, before=before, after=after,
                production_evaluations=layer._evaluation.misses,
                best=[r[0] for r in layer._cache[('cold',)]])


def isolated_benchmark(index: int, row: dict, overlay: object) -> dict:
    """他ジョブが重なった標本だけを棄却し、同じ事前固定indexを再試行する。"""
    while True:
        while idle_snapshot()['other_python']:
            time.sleep(POLL_SEC)
        try:
            return benchmark(index, row, overlay)
        except MeasurementBusy as error:
            print(str(error), flush=True)
            time.sleep(POLL_SEC)


def main() -> None:
    """診断は範囲分割可。正式測定は他ジョブ終了を待ち、採点前にLを保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('diagnose', 'benchmark'))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=100)
    args = parser.parse_args()
    rows = json.loads((reference.OUT/'samples_ledger.json').read_text())
    if args.mode == 'benchmark':
        while idle_snapshot()['other_python']:
            time.sleep(POLL_SEC)
    overlay = reference.model_overlay()
    indices = replay.TIMING_INDICES if args.mode == 'benchmark' else range(args.start, args.stop)
    for index in indices:
        path = OUT/('notifications' if args.mode == 'benchmark' else 'diagnostics_full')/f'{args.mode}_{index:03d}.json'
        if path.exists():
            continue
        snapshot(overlay, rows[index])
        result = isolated_benchmark(index, rows[index], overlay) if args.mode == 'benchmark' else diagnose(index, rows[index], overlay)
        save(path, result)
        print(args.mode, index, flush=True)
    if args.mode == 'benchmark':
        replay.OUT = OUT
        print(replay.freeze_latency(), flush=True)


if __name__ == '__main__':
    main()
