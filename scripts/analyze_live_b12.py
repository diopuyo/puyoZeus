"""外部CPU負荷の汚染区間を含む/除く両方の母数付き集計。"""
from __future__ import annotations

from bisect import bisect_right
import json
import math
from pathlib import Path

import numpy as np

from scripts.analyze_live_b9 import lines, samples, available, quantile
from scripts.measure_live_b6 import save

FPS = 30
WINDOW_SECONDS = 300
MS = 1000
TIMESTAMP_DIGITS = 6


class Contamination:
    def __init__(self, rows: list[dict], field: str = 'contaminated') -> None:
        self.rows = sorted(rows, key=lambda row: row['start'])
        self.starts = [row['start'] for row in self.rows]
        self.field = field

    def label(self, start: float, end: float | None = None) -> str:
        end = start if end is None else end
        index = bisect_right(self.starts, start)-1
        cursor, contaminated = start, False
        while 0 <= index < len(self.rows):
            row = self.rows[index]
            if row['start'] > cursor or row['end'] <= cursor or not row['valid'] or self.field not in row:
                return 'unknown'
            contaminated |= row[self.field]
            if row['end'] >= end:
                return 'contaminated' if contaminated else 'clean'
            cursor, index = row['end'], index+1
        return 'unknown'


def aggregate(slots: list[dict], sent: list[dict], resources: list[dict], clean: bool) -> dict:
    selected = [r for r in slots if not clean or r['label'] == 'clean']
    rows = [r for r in sent if not clean or r['label'] == 'clean']
    memory = [r['rss_bytes'] for r in resources if not clean or r['label'] == 'clean']
    drops = sum(r['dropped'] for r in selected)
    return dict(expected=len(selected), dropped=drops,
        drop_fraction=drops/len(selected) if selected else None,
        latency_samples=len(rows), latency_p95_ms=quantile([r['latency_ms'] for r in rows]),
        rss_samples=len(memory), rss_p50_bytes=quantile(memory, 50))


def report(path: Path, start: float, end: float, unplanned: bool = False) -> dict:
    mask = Contamination(lines(path/'cpu_load.jsonl'), 'unplanned_contaminated' if unplanned else 'contaminated')
    with np.load(path/'recognition.npz') as data:
        origin = float(np.median(data['captured_at']-data['t_sec']))
    drops = {round(t, TIMESTAMP_DIGITS) for t in json.loads((path/'recognition.json').read_text())['dropped_times']}
    slots = [dict(t_sec=t, dropped=round(t, TIMESTAMP_DIGITS) in drops, label=mask.label(origin+t))
             for t in slot_times(path, start, end)]
    expected_drops = json.loads((path/'metrics.json').read_text())['capture_dropped_in_measured_window']
    if sum(r['dropped'] for r in slots) != expected_drops:
        raise ValueError('計測器と汚染集計の捨てframe件数が一致しません')
    sent = []
    for row in samples(path):
        if available(row):
            timing = row['payload']['timing']
            captured = timing['capture_monotonic_sec']
            sent.append(dict(t_sec=timing['source_available_ms']/MS,
                latency_ms=(row['received_at']-captured)*MS,
                label=mask.label(captured, row['received_at'])))
    resources = [dict(r, label=mask.label(r['at'])) for r in lines(path/'resources.jsonl')]
    output = []
    for lo in np.arange(start, end, WINDOW_SECONDS):
        hi = min(lo+WINDOW_SECONDS, end)
        selected = [r for r in slots if lo <= r['t_sec'] < hi]
        ss = [r for r in sent if lo <= r['t_sec'] < hi]
        rr = [r for r in resources if origin+lo <= r['at'] < origin+hi]
        output.append(dict(start=float(lo), end=float(hi),
            labels={key: sum(r['label'] == key for r in selected) for key in ('clean', 'contaminated', 'unknown')},
            including=aggregate(selected, ss, rr, False), excluding=aggregate(selected, ss, rr, True)))
    result = dict(windows=output, intervals=mask.rows,
                  note='外部>1 CPUコアを汚染。欠測はcleanに含めない。SSEはcapture〜受信の全区間で分類。')
    save(path/('unplanned_contamination.json' if unplanned else 'contamination.json'), result)
    return result


def normalized_slots(bounds: dict, start: float) -> list[float]:
    fps, first, end, stride = (bounds[key] for key in ('fps', 'start', 'end', 'stride'))
    first += max(0, math.ceil((start*fps-first)/stride))*stride
    return [frame/fps for frame in range(first, end, stride)]


def slot_times(path: Path, start: float, end: float) -> list[float]:
    """60fps入力の奇数/偶数位相を維持する。秒数×30の丸めでは1frameずれる。"""
    metrics = json.loads((path/'metrics.json').read_text())
    bounds = metrics.get('frame_bounds')
    if bounds is None:
        with np.load(path/'recognition.npz') as data:
            valid = data['t_sec'] > 0
            fps = round(float(np.median(data['frame'][valid]/data['t_sec'][valid])))
            stride = round(fps/FPS)
            bounds = dict(fps=fps, start=int(data['frame'][0]) % stride, end=int(end*fps), stride=stride)
    times = normalized_slots(bounds, start)
    if len(times) != metrics['expected_frames']:
        raise ValueError('計測器と汚染集計の対象frame母数が一致しません')
    return times


def compare(reference: Path, path: Path, start: float, end: float) -> dict:
    from scripts.analyze_live_b8 import board_comparison, load_audit, summarize_recognition
    bounds = json.loads((path/'metrics.json').read_text()).get('frame_bounds')
    actual_end = min(end, bounds['end']/bounds['fps']) if bounds else end
    boards = board_comparison(load_audit(reference/'recognition.npz', start, actual_end),
                              load_audit(path/'recognition.npz', start, actual_end))
    result = dict(boards=boards, contamination=report(path, start, end),
                  unplanned=report(path, start, end, unplanned=True),
                  requested_end_sec=end, reference_end_sec=actual_end)
    result['placement_intervals'] = placement_intervals(reference, path, start, actual_end, boards)
    result['recognition_cost'] = summarize_recognition(path, start, end)
    result['scheduling'] = scheduling(path, start, end)
    if (reference/'events.jsonl').exists():
        result['physics'] = compare_physics(reference, path, start, actual_end)
    save(path/'comparison.json', result)
    return result


def scheduling(path: Path, start: float, end: float) -> dict:
    """表示頻度低下をcapture→SSEの短縮と混同しないよう、実更新間隔も残す。"""
    from collections import Counter
    metrics = json.loads((path/'metrics.json').read_text())
    counts = Counter(r['period_sec'] for r in metrics['probability_calculations'] if start <= r['t_sec'] < end)
    rows = [r for r in samples(path) if available(r) and
            start <= r['payload']['timing']['source_available_ms']/MS < end]
    gaps = [b['received_at']-a['received_at'] for a, b in zip(rows, rows[1:])
            if a['payload']['identity']['match_id'] == b['payload']['identity']['match_id']]
    return dict(period_counts=dict(counts), display_gap_samples=len(gaps),
                display_gap_p95_sec=quantile(gaps), cpu_runtime=metrics['cpu_runtime'])


def placement_intervals(reference: Path, path: Path, start: float, end: float, boards: dict) -> dict:
    """置き対応の探索範囲全体がcleanのものだけを、除外後の母数へ入れる。"""
    from scripts.analyze_live_b8 import load_audit, board_events, placement_indices, MAX_SHIFT_SEC
    left = load_audit(reference/'recognition.npz', start, end)
    with np.load(path/'recognition.npz') as data:
        origin = float(np.median(data['captured_at']-data['t_sec']))
    masks = {key: Contamination(lines(path/'cpu_load.jsonl'), key)
             for key in ('contaminated', 'unplanned_contaminated')}
    rows = []
    for side in range(2):
        sequence = board_events(left, side)
        missing = boards['sides'][side]['placements']
        for i in placement_indices(sequence):
            t_sec = float(left['t_sec'][sequence[i]['indices'][0]])
            status = ('omitted' if i in missing['omitted_indices'] else
                      'unresolved' if i in missing['unresolved_indices'] else 'matched')
            labels = {key: mask.label(origin+t_sec-MAX_SHIFT_SEC, origin+t_sec+MAX_SHIFT_SEC)
                      for key, mask in masks.items()}
            rows.append(dict(side=side+1, t_sec=t_sec, status=status, **labels))
    result = dict(rows=rows, including=placement_counts(rows))
    for key in masks:
        result['excluding_'+key] = placement_counts([r for r in rows if r[key] == 'clean'])
    return result


def placement_counts(rows: list[dict]) -> dict:
    return dict(reference_placements=len(rows),
                **{key: sum(r['status'] == key for r in rows) for key in ('matched', 'omitted', 'unresolved')})


def compare_physics(reference: Path, path: Path, start: float, end: float) -> dict:
    from scripts.analyze_live_b8 import read_physics, match_times, timing_result
    data = [read_physics(p/'events.jsonl') for p in (reference, path)]
    chains = [[r for r in items[0] if start <= r['trigger_sec'] < end] for items in data]
    chains = [[dict(r, **{key: r.get(key) if r.get(key) is not None and start <= r[key] < end else None
                         for key in ('observed_sec', 'end_signal_sec')}) for r in side] for side in chains]
    landings = [[r for r in items[1] if start <= r['t_sec'] < end] for items in data]
    pairs = match_times(*chains, 'trigger_sec')
    return dict(chain_start=timing_result(*chains, pairs, 'observed_sec'),
        chain_end=timing_result(*chains, pairs, 'end_signal_sec'),
        landing=timing_result(*landings, match_times(*landings, 't_sec'), 't_sec'))
