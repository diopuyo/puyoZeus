"""保存済み判定列を仮想時計で送信経路へ流す。認識・推論・OBSは起動しない。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from src.phase_j import live_publish as delivery

MS = 1000.0
PERCENTILES = (50, 95, 99)
ASSETS = {name: 'saved-replay' for name in ('app_build_id', 'recognition_model_hash',
    'recognition_config_hash', 'prediction_model_hash', 'calibration_hash')}


@dataclass
class ReplayClock:
    """実経路のEvent待機とsleepを、保存時刻の離散イベントへ置き換える。"""

    events: list[tuple[float, dict]]
    now: float
    end: float
    publisher: Any = None
    cursor: int = 0
    signaled: bool = False
    offered: list[dict] = field(default_factory=list)

    def advance(self, target: float) -> None:
        while self.cursor < len(self.events) and self.events[self.cursor][0] <= target:
            stamp, row = self.events[self.cursor]
            self.now = stamp
            self.publisher.offer(row)
            self.offered.append(row)
            self.cursor += 1
        self.now = target

    def wait(self, timeout: float | None = None) -> bool:
        if self.signaled:
            return True
        target = self.end if timeout is None else min(self.end, self.now + timeout)
        if self.cursor < len(self.events):
            target = min(target, self.events[self.cursor][0])
        self.advance(target)
        return self.signaled or self.now >= self.end

    def set(self) -> None:
        self.signaled = True

    def clear(self) -> None:
        self.signaled = False

    def sleep(self, seconds: float) -> None:
        self.advance(self.now + seconds)


def load_inputs(directory: Path) -> tuple[list[dict], dict, float]:
    rows = json.loads((directory / 'evaluations.json').read_text(encoding='utf-8'))
    updates = [json.loads(line) for line in
               (directory / 'b3_updates.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted = {(r['game'], r['frame']): r for r in updates
                if r.get('accepted', True) and r['error'] is None}
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    return rows, accepted, float(metrics['frame_bounds']['fps'])


def replay_inputs(rows: list[dict], updates: dict, immediate: bool) -> list[tuple[float, dict]]:
    """値は保存列を保持。新経路の時刻だけを当該B3結果の完了時刻へ移す。"""
    events = []
    last_frames: dict[int, int] = {}
    for original in rows:
        row = deepcopy(original)
        game = row['game']
        frame = row.get('b3_state_frame', last_frames.get(game, -1))
        last_frames[game] = frame
        update = updates.get((game, frame))
        matched = update is not None and update['raw_probability'] == row['raw_probability']
        complete = update['completed_at'] if matched else row['evaluated_at']
        row['judgment_id'] = (game, frame, row['source'], row['raw_probability'], row['match_active'])
        row['completed_at'] = complete
        stamp = row['evaluated_at']
        if immediate and matched:
            stamp = complete
            row.update(frame=frame, t_sec=update['t_sec'], captured_at=update['captured_at'],
                       recognized_at=complete, evaluated_at=complete)
        events.append((stamp, row))
    return sorted(events, key=lambda event: event[0])


def run_delivery(events: list[tuple[float, dict]], start: float, end: float,
                 immediate: bool) -> tuple[list[dict], list[dict]]:
    """本番のPublisher・DTO・SSE書込みを使用し、ソケットと壁時計だけを置換する。"""
    publisher = delivery.LivePublisher(ASSETS, '127.0.0.1', 0,
                                       evaluation_driven=True, completion_driven=immediate)
    clock = ReplayClock(events, start, end, publisher)
    publisher.ready = clock
    publisher.stop = SimpleNamespace(is_set=lambda: clock.now >= end)
    handler_type = publisher.server._make_handler_class()
    handler = handler_type.__new__(handler_type)
    handler.wfile = BytesIO()
    original_publish = publisher.state.publish
    sent: list[dict] = []

    def publish(snapshot: Any) -> None:
        original_publish(snapshot)
        key = tuple(publisher.latest['judgment_id'])
        before = handler.wfile.tell()
        handler._write_sse('analysis', snapshot.to_mapping())
        if handler.wfile.tell() != before:
            sent.append(dict(at=clock.now, key=key, payload=snapshot.to_mapping()))

    publisher.state.publish = publish
    virtual_time = SimpleNamespace(perf_counter=lambda: clock.now, sleep=clock.sleep)
    with patch.object(delivery, 'time', virtual_time):
        publisher._run()
    if publisher.error is not None:
        raise publisher.error
    assert len(clock.offered) == len(events)
    return sent, clock.offered


def summary(sent: list[dict], events: list[tuple[float, dict]], fps: float) -> dict:
    """送られなかった判定も母数へ残す。遅れは同じ判定の最初の送信まで。"""
    completions = {}
    for _, row in events:
        if row['raw_probability'] is not None and row['match_active']:
            key = tuple(row['judgment_id'])
            completions.setdefault(key, row['completed_at'])
    first_sent = {}
    repeats = 0
    for record in sent:
        payload = record['payload']
        practical = payload['evaluations']['practical']
        key = record['key']
        if key not in completions or practical['availability'] != 'available':
            continue
        repeats += int(key in first_sent)
        first_sent.setdefault(key, record['at'])
    delays = [(stamp-completions[key])*MS for key, stamp in first_sent.items()]
    return dict(sends=len(sent), duplicate_judgment_sends=repeats,
                judgments=len(completions), delivered=len(first_sent),
                delay_ms=percentile(delays), delay_frames=percentile([d*fps/MS for d in delays]))


def percentile(values: list[float]) -> dict:
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                **{f'p{p}': float(np.percentile(values, p)) if values else None for p in PERCENTILES})


def judgment_values(row: dict) -> tuple:
    """入力の生値と、公開DTOへ実際に投影された未平滑の値を照合する。"""
    snapshot = delivery.result_snapshot(delivery.initial_snapshot(ASSETS), row,
                                         row['evaluated_at'], 1, None)
    layers = snapshot.evaluations.get('display_layers', {})
    return row['raw_probability'], row['source'], layers.get('current_p1')


def verify(directory: Path) -> dict:
    started = perf_counter()
    rows, updates, fps = load_inputs(directory)
    old_events = replay_inputs(rows, updates, False)
    new_events = replay_inputs(rows, updates, True)
    start = min(old_events[0][0], new_events[0][0])
    end = max(old_events[-1][0], new_events[-1][0]) + delivery.PUBLISH_PERIOD_SEC
    before, old_offered = run_delivery(old_events, start, end, False)
    after, new_offered = run_delivery(new_events, start, end, True)
    identity = lambda row: (row['game'], row['judgment_id'], row['digest'])
    old_values = {identity(row): judgment_values(row) for row in old_offered}
    equal = sum(old_values[identity(row)] == judgment_values(row) for row in new_offered)
    assert equal == len(rows)
    return dict(input=str(directory), basis='保存判定列・仮想時計 (IPC/OS/OBS実遅延は含まない)',
                rows=len(rows), judgment_equal=equal, fps=fps, window_sec=end-start,
                before=summary(before, old_events, fps), after=summary(after, new_events, fps),
                elapsed_sec=perf_counter()-started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = verify(options.input)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
