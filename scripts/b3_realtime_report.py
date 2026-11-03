"""B3更新原票と実SSE送信を対応させ、時間門の母数と欠落を報告する。"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np

MS = 1000.0
PLACEMENT = 1
MINUTE_SEC = 60
STAGES = ('state', 'ledger', 'search', 'feature_only', 'inference')
WAITS = ('state_wait_ms', 'compute_wait_ms', 'result_wait_ms', 'wait_ms')


def percentiles(values: list[float]) -> dict:
    if not values:
        return dict(n=0)
    result = dict(zip(('p50', 'p95', 'p99', 'max'),
                      map(float, np.percentile(values, (50, 95, 99, 100)))))
    return dict(n=len(values), **result)


def waiting(metrics: dict, first_frame: int) -> dict:
    """取得予定時刻を起点に、認識込みの未処理待ちと毎秒の滞留を測る。"""
    captures = {r['frame']: r['captured_at'] for r in metrics['recognition']}
    rows = [r for r in metrics['evaluation_profile'] if r['frame'] in captures]
    waits = [(r['started_at']-captures[r['frame']])*MS for r in rows]
    if not metrics['realtime']:
        return dict(wait_ms=percentiles(waits), wait_over_1s=sum(v > MS for v in waits),
                    note='無ペーシング試験。予定時刻ベースの30分末滞留・drainは算出しない。')
    bounds = metrics['frame_bounds']
    start = min(t-(f-first_frame)/bounds['fps'] for f, t in captures.items()) if captures else 0
    duration = (bounds['end']-first_frame)/bounds['fps']
    samples = []
    for second in range(int(duration)+1):
        stamp = start+second
        pending = sum(captures[r['frame']] <= stamp < r['finished_at'] for r in rows)
        samples.append(dict(second=second, pending=pending))
    minutes = [dict(minute=i, **percentiles([s['pending'] for s in samples
        if i*MINUTE_SEC <= s['second'] < (i+1)*MINUTE_SEC]))
        for i in range(int(np.ceil(duration/MINUTE_SEC)))]
    return dict(wait_ms=percentiles(waits), wait_over_1s=sum(v > MS for v in waits),
        pending_including_processing=samples, pending_at_deadline=samples[-1]['pending'],
        queue_minutes=minutes, duration_sec=duration,
        drain_after_deadline_ms=max(0.0, max((r['finished_at'] for r in rows), default=start)-start-duration)*MS,
        note='取得済み通知の認識待ち＋評価待ち＋処理中。capture dropは別母数。')


def breakdown(directory: Path, updates: list[dict], first: int, end: int) -> dict:
    """計算採否に関係なく原票を集計。状態原票の母数は全観測。"""
    path = directory/'b3_state_timing.jsonl'
    states = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    states = [r for r in states if first <= r['frame'] < end]
    keys = tuple(stage+suffix for stage in STAGES for suffix in ('_ms', '_cpu_ms')) + WAITS
    def stats(rows: list[dict], names: tuple[str, ...]) -> dict:
        return {key: dict(percentiles([r[key] for r in rows if key in r]),
                          total=sum(r.get(key, 0.) for r in rows)) for key in names}
    return dict(computed=stats(updates, keys), all_observations=stats(states,
        ('state_ms', 'state_cpu_ms', 'ledger_ms', 'ledger_cpu_ms', 'search_ms', 'search_cpu_ms', 'state_wait_ms')),
        observations=len(states), requested=sum(bool(r.get('trigger', 0)) for r in states),
        accepted=sum(r.get('accepted', True) for r in updates),
        rejected=sum(not r.get('accepted', True) for r in updates))


def placement_latencies(metrics: dict, placements: list[dict], asynchronous: bool) -> tuple[list, int]:
    """非同期では旧SSEフレーム対応を使わない。結果の元フレームとの照合が必要。"""
    if asynchronous:
        return [], 0
    first: dict[int, float] = {}
    for row in metrics['sse_sent']:
        if row.get('available', True):
            frame = int(row['frame'])
            first[frame] = min(first.get(frame, float('inf')), row['sent_at'])
    frames = sorted(first)
    latencies, coalesced = [], 0
    for row in placements:
        i = bisect.bisect_left(frames, row['frame'])
        if i < len(frames):
            latencies.append((first[frames[i]]-row['captured_at'])*MS)
            coalesced += frames[i] != row['frame']
    return latencies, coalesced


def report(directory: Path) -> dict:
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    updates = [json.loads(line) for line in (directory/'b3_updates.jsonl').read_text().splitlines()]
    fatal_errors = [r['fatal'] for r in updates if 'fatal' in r]
    config = directory/'b3_config.json'
    asynchronous = config.exists() and json.loads(config.read_text()).get('async_enabled', False)
    bounds = metrics['frame_bounds']
    first_frame = bounds['end']-metrics['expected_frames']*bounds['stride']
    updates = [r for r in updates if 'frame' in r and first_frame <= r['frame'] < bounds['end']]
    placements = [r for r in updates if r['trigger'] & PLACEMENT]
    latencies, coalesced = placement_latencies(metrics, placements, asynchronous)
    return dict(updates=len(updates), inference_errors=sum(r['error'] is not None for r in updates),
        async_enabled=asynchronous, worker_errors=fatal_errors,
        breakdown=breakdown(directory, updates, first_frame, bounds['end']),
        feature_ms=percentiles([r['feature_ms'] for r in updates]),
        inference_ms=percentiles([r['inference_ms'] for r in updates]),
        placement_to_sse_ms=percentiles(latencies), placements=len(placements),
        placement_without_later_sse=None if asynchronous else len(placements)-len(latencies),
        coalesced_placements=None if asynchronous else coalesced,
        queue=metrics['evaluation_queue'], capture_drops=metrics['capture_dropped_in_measured_window'],
        recognition_frames=metrics['recognition_frames_including_warmup'],
        expected_frames=metrics['expected_frames'], bounds=metrics['frame_bounds'],
        waiting=waiting(metrics, first_frame), measured_first_frame=first_frame,
        scope='PC1の本番認識+B3+R死亡証明+SSE。ブラウザ実描画は別途OBS試験。',
        timing_basis=('非同期の設置→SSEは未集計。SSEのpractical.request_idと更新元frameを照合する。'
                      if asynchronous else '取得予定壁時計→その設置以降の初回SSE flush。間引き母数は別記。'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    (args.directory/'b3_timing.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['queue'].pop('observed', None)
    result['waiting'].pop('pending_including_processing', None)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
