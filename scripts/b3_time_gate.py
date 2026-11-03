"""B3固定行生成器の因果的再生。特徴と最終モデルの時間だけ測り、採点しない。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from src import b3_rows, b3_search
from src.b3_source import iter_frames, load_events, load_lean, resolve_attempt, sha256_file
from src.b3_state import GameState
from src.b3_timing import TimingTable
from src.b3_train_model import load_model
from scripts.b3_time_gate_support import Instrumentation, MS, environment, memory, search_status, summarize, write_json

ROOT = Path('/mnt/d/puyo_analyzer')
MODEL = ROOT / 'share/results/pc2/final_mono/b3_final_mono_5ea0474.joblib'
TABLE = ROOT / 'b3_data/timing_v1/timing_table.json'
START_SEC = 300.0
DURATION_SEC = 1800.0
MONITOR_SEC = 1.0
WAIT_SLICE_SEC = 0.05
STARTUP_SEC = 5.0


def dump_line(stream: Any, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


class Replay:
    """generate_rowsのフレームループと同じ状態・イベント・更新契機を使う。"""

    def __init__(self, record: Path, video: str, table: TimingTable) -> None:
        self.gs = GameState(load_lean(record))
        self.events = load_events(record)
        self.placed = b3_rows._placement_frames(self.events)
        self.idx = 0
        self.boundary = False
        self.last_t = -float('inf')
        self.prev_icon: list = [None, None]
        self.prev_land = [0, 0]
        self.video, self.table = video, table
        manifest = ROOT / 'b3_data/eval_rows_v2b' / (video + '.manifest.json')
        self.mapping = json.loads(manifest.read_text())['winner_mapping']

    def step(self, d: dict, make_features: bool = True) -> tuple[dict | None, float]:
        start = time.perf_counter()
        self.boundary |= self.gs.update_frame(d, self.placed.get(d['frame'], ()))
        trig = 0
        while self.idx < len(self.events) and self.events[self.idx]['frame'] <= d['frame']:
            trig |= self.gs.apply_event(self.events[self.idx])
            self.idx += 1
        if not self.gs.in_match:
            return None, time.perf_counter() - start
        trig |= b3_rows._frame_triggers(self.gs, self.boundary, self.last_t, b3_rows.HEARTBEAT_SEC,
                                        self.prev_icon, self.prev_land)
        self.boundary = False
        state_sec = time.perf_counter() - start
        if trig == 0 or all(s.board is None for s in self.gs.sides):
            return None, state_sec
        self.last_t = self.gs.t
        if not make_features or not b3_rows._inside_match(self.gs.t, self.mapping.get(str(self.gs.game_idx), {})):
            return None, state_sec
        return b3_rows._make_row(self.gs, self.table, self.video, trig, None), state_sec


def producer(record: str, start: float, duration: float, origin: float, tasks: Any,
             submitted: Any, completed: Any, output: str) -> None:
    """独立プロセスが固定時刻で投入する。消費側の速さによるbackpressureをかけない。"""
    with Path(output).open('w', buffering=1) as stream:
        next_monitor = origin
        for d in iter_frames(Path(record)):
            source = float(d['available_at_sec'])
            if source < start:
                continue
            if source >= start + duration:
                break
            due = origin + source - start
            while time.perf_counter() < due:
                now = time.perf_counter()
                if now >= next_monitor:
                    dump_line(stream, dict(wall_sec=now-origin, submitted=submitted.value,
                                            completed=completed.value, queue=submitted.value-completed.value))
                    next_monitor += MONITOR_SEC
                time.sleep(min(WAIT_SLICE_SEC, max(0.0, due-now)))
            tasks.put((d, due, time.perf_counter()))
            with submitted.get_lock():
                submitted.value += 1
        remaining = origin + duration - time.perf_counter()
        while remaining > 0:
            time.sleep(min(WAIT_SLICE_SEC, remaining))
            remaining = origin + duration - time.perf_counter()
        dump_line(stream, dict(wall_sec=duration, submitted=submitted.value,
                                completed=completed.value, queue=submitted.value-completed.value, final=True))
        tasks.put(None)


def resources(stop: threading.Event, path: Path, origin: float) -> None:
    """計測中は1秒周期で空きメモリを観測する。"""
    with path.open('w', buffering=1) as stream:
        while not stop.is_set():
            dump_line(stream, dict(wall_sec=time.perf_counter()-origin, **memory()))
            stop.wait(MONITOR_SEC)
        dump_line(stream, dict(wall_sec=time.perf_counter()-origin, **memory()))


def process_frame(replay: Replay, model: Any, inst: Instrumentation, item: tuple,
                  outputs: dict, cumulative: dict, origin: float, realtime: bool) -> None:
    """処理開始から状態更新・特徴・1行推論・flushによる公開までを計測する。"""
    d, due, offered = item
    inst.reset()
    before = b3_search._search_cached.cache_info()
    begin = time.perf_counter()
    row, state_sec = replay.step(d)
    feature_end = time.perf_counter()
    feature_ms = (feature_end-begin) * MS
    cumulative['feature_ms'] += feature_ms
    wait_ms = max(0.0, begin-due) * MS if realtime else 0.0
    frame = dict(frame=d['frame'], t=d['t_sec'], wait_ms=wait_ms,
                 ingest_lag_ms=(offered-due)*MS if realtime else 0.0, feature_ms=feature_ms,
                 ledger_ms=inst.seconds['ledger']*MS, state_ms=state_sec*MS)
    if row is not None:
        inference_start = time.perf_counter()
        probability = model.predict_row(row)
        inference_end = time.perf_counter()
        if not np.isfinite(probability):
            raise ValueError('有限な勝率を公開できない')
        dump_line(outputs['publications'], dict(frame=d['frame'], t=d['t_sec'], p=probability))
        outputs['publications'].flush()
        published = time.perf_counter()
        after = b3_search._search_cached.cache_info()
        result = dict(video=replay.video, game=d['game_idx'], frame=d['frame'], t=d['t_sec'],
                      available_at=d['available_at_sec'], trigger=row['trigger_mask'],
                      placement=bool(row['trigger_mask'] & b3_rows.TRIGGER_BITS['placement']),
                      feature_ms=feature_ms, state_ms=state_sec*MS, row_ms=feature_ms-state_sec*MS,
                      inference_ms=(inference_end-inference_start)*MS,
                      latency_ms=(published-(due if realtime else begin))*MS, wait_ms=wait_ms,
                      wall_sec=published-origin, amortized_feature_ms=cumulative['feature_ms'],
                      cache_hits=after.hits-before.hits, cache_misses=after.misses-before.misses,
                      search_calls=inst.calls['search'], **search_status(row))
        result.update({k+'_ms': inst.seconds[k]*MS for k in ('ledger', 'search', 'indicators')})
        dump_line(outputs['updates'], result)
        cumulative.clear()
    dump_line(outputs['frames'], frame)


def run_realtime(replay: Replay, model: Any, inst: Instrumentation, record: Path,
                 out: Path, outputs: dict, args: argparse.Namespace, origin: float) -> dict:
    """区間前を因果的に畳み込み、区間は1800秒の実時間投入。全キューを最後まで処理する。"""
    context = mp.get_context('spawn')
    tasks = context.Queue()
    submitted, completed = context.Value('q', 0), context.Value('q', 0)
    worker = context.Process(target=producer, args=(str(record), args.start, args.duration, origin,
                              tasks, submitted, completed, str(out / 'queue.jsonl')))
    worker.start()
    cumulative: dict = defaultdict(float)
    while (item := tasks.get()) is not None:
        process_frame(replay, model, inst, item, outputs, cumulative, origin, True)
        with completed.get_lock():
            completed.value += 1
    worker.join()
    if worker.exitcode != 0:
        raise RuntimeError(f'投入プロセス失敗: {worker.exitcode}')
    return dict(submitted=submitted.value, completed=completed.value,
                drain_wall_sec=time.perf_counter()-origin)


def run_unpaced(replay: Replay, model: Any, inst: Instrumentation, record: Path,
                outputs: dict, origin: float, args: argparse.Namespace) -> dict:
    """全観測を順番どおり処理。入力は消費に追随し、スループットだけを測る。"""
    cumulative: dict = defaultdict(float)
    count = 0
    for d in iter_frames(record):
        if args.limit_frames is not None and count >= args.limit_frames:
            break
        now = time.perf_counter()
        process_frame(replay, model, inst, (d, now, now), outputs, cumulative, origin, False)
        count += 1
    return dict(frames=count, wall_sec=time.perf_counter()-origin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('realtime', 'unpaced'), required=True)
    parser.add_argument('--video', choices=('b3_zenchi_s1', 'b3_zenchi_s2'), default='b3_zenchi_s1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=float, default=START_SEC)
    parser.add_argument('--duration', type=float, default=DURATION_SEC)
    parser.add_argument('--limit-frames', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    record = resolve_attempt(args.video, ROOT / 'b3_eval_records').record
    model = load_model(MODEL)
    table = TimingTable.from_json(TABLE.read_text())
    replay = Replay(record, args.video, table)
    metadata = dict(environment=environment(), args={k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()},
                    model_sha256=sha256_file(MODEL), timing_sha256=sha256_file(TABLE),
                    input_sha256={n: sha256_file(record / n) for n in
                                  ('observations.jsonl.gz', 'events.jsonl', 'lean.npz')},
                    heartbeat=b3_rows.HEARTBEAT_SEC, monitor_sec=MONITOR_SEC,
                    scope='B3特徴+推論。CNN認識・OBS描画・R死亡証明は測定外。勝敗は読まず全match_activeを処理。')
    if args.mode == 'realtime':
        for d in iter_frames(record):
            if d['available_at_sec'] >= args.start:
                break
            replay.step(d, make_features=False)
        metadata['prefix'] = '状態だけを開始時刻まで畳み込む。特徴探索cacheはcold開始。'
    write_json(args.out / 'metadata.json', metadata)
    inst = Instrumentation()
    inst.install()
    # 子プロセスの起動時間を投入時刻へ混入させないため5秒の準備時間を置く。
    origin = time.perf_counter() + (STARTUP_SEC if args.mode == 'realtime' else 0.0)
    stop = threading.Event()
    monitor = threading.Thread(target=resources, args=(stop, args.out / 'resources.jsonl', origin))
    monitor.start()
    outputs = {n: (args.out / (n+'.jsonl')).open('w', buffering=1)
               for n in ('updates', 'frames', 'publications')}
    try:
        run = run_realtime if args.mode == 'realtime' else run_unpaced
        if args.mode == 'realtime':
            result = run(replay, model, inst, record, args.out, outputs, args, origin)
        else:
            result = run(replay, model, inst, record, outputs, origin, args)
        write_json(args.out / 'completion.json', result)
    finally:
        stop.set()
        monitor.join()
        inst.restore()
        for stream in outputs.values():
            stream.close()
    print(json.dumps(summarize(args.out), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
