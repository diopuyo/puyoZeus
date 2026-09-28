"""保存済みB14/B16の同じ物差しを用いた段別・負荷・履歴の診断。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analyze_live_b9 import available, lines, samples
from scripts.measure_live_b6 import save

START, WINDOW, MS, FPS = 2580.566, 300.0, 1000.0, 30.0
ROOT = Path('logs/live_b17')
PATHS = {'b14': Path('logs/live_b14/long'), 'b16': Path('logs/live_b16/realtime')}


def percentiles(values: Any) -> dict:
    array = np.asarray(values)
    return dict(count=len(array), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95))) if len(array) else dict(count=0)


def delivery(rows: list[dict]) -> dict:
    values = dict(capture_recognition=[], recognition_evaluation=[], evaluation_sse=[], capture_sse=[])
    for row in rows:
        timing = row['payload']['timing']
        capture, recognized, evaluated = (timing[k+'_monotonic_sec']
                                          for k in ('capture', 'recognized', 'evaluated'))
        for key, elapsed in zip(values, (recognized-capture, evaluated-recognized,
                                        row['received_at']-evaluated, row['received_at']-capture)):
            values[key].append(elapsed*MS)
    return {key: percentiles(value) for key, value in values.items()}


def window(path: Path, metrics: dict, data: Any, sent: list, lo: float, hi: float) -> dict:
    origin = float(np.median(data['captured_at']-data['t_sec']))
    mask = (data['t_sec'] >= lo) & (data['t_sec'] < hi)
    selected = [r for r in sent if lo <= r['payload']['timing']['source_available_ms']/MS < hi]
    load = [r for r in lines(path/'cpu_load.jsonl') if origin+lo <= r['start'] < origin+hi]
    updates = [r for r in metrics['state_updates'] if lo <= r['t_sec'] < hi]
    calculations = [r for r in metrics['probability_calculations'] if lo <= r['t_sec'] < hi]
    with (path/'review_data.csv').open() as stream:
        review = [r for r in csv.DictReader(stream) if lo <= float(r['t_sec']) < hi]
    return dict(start=lo, end=hi, delivery_ms=delivery(selected),
        recognition_ms=percentiles(data['recognition_ms'][mask]),
        recognition_cpu_ms=percentiles(data['cpu_ms'][mask]),
        acquisition_wait_ms=percentiles((data['acquired_at'][mask]-data['captured_at'][mask])*MS),
        queue_put_ms=percentiles(data['queue_put_ms'][mask]),
        state_update_ms=percentiles([r['milliseconds'] for r in updates]),
        calculation_ms=percentiles([r['milliseconds'] for r in calculations]),
        queue_depth=percentiles([r['queue_depth'] for r in updates]),
        external_cores=percentiles([r['external_cores'] for r in load]),
        contaminated_seconds=sum(r['elapsed_sec'] for r in load if r['contaminated']),
        reviewed_publications=len(review),
        nonstable_publications=sum(r['1P_state'] != 'STABLE' or r['2P_state'] != 'STABLE' for r in review))


def analyze(path: Path, start: float = START) -> dict:
    metrics = json.loads((path/'metrics.json').read_text())
    sent = [r for r in samples(path) if available(r)]
    with np.load(path/'recognition.npz') as data:
        end = float(data['t_sec'][-1])+1/FPS
        output = [window(path, metrics, data, sent, float(lo), float(min(lo+WINDOW, end)))
                  for lo in np.arange(start, end, WINDOW)]
    notifications = np.diff([0]+[r['notifications'] for r in metrics['probability_calculations']])
    report = path.parent/'report.json'
    rss = json.loads(report.read_text()).get('rss_after_first_5min') if report.exists() else None
    return dict(path=str(path), windows=output, batch_notifications=percentiles(notifications),
        queue={k:v for k,v in metrics['evaluation_queue'].items() if k != 'observed'},
        dropped=metrics['capture_dropped_in_measured_window'], expected=metrics['expected_frames'],
        rss=rss)


def main() -> None:
    from scripts.measure_live_b17 import START as short_start
    paths = {key: (path, START) for key, path in PATHS.items()}
    for mode in ('baseline', 'candidate'):
        path = ROOT/'live'/mode
        if (path/'metrics.json').exists():
            paths[mode] = (path, short_start)
    result = dict(definition='availableかつstream_seq重複除外。予定captureからSSE受信まで。',
                  runs={key: analyze(path, start) for key, (path, start) in paths.items()})
    save(ROOT/'saved_diagnosis.json', result)
    print(json.dumps({key: dict(dropped=r['dropped'], expected=r['expected'],
        latency_p95_ms=[w['delivery_ms']['capture_sse']['p95'] for w in r['windows']])
        for key, r in result['runs'].items()}))


if __name__ == '__main__':
    main()
