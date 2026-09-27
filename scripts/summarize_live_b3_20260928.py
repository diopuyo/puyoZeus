"""B3スモークの母数付き比較・待ち時間内訳をJSONへまとめる。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_live_pipeline_20260928 import compare_arrays, percentile, save_json

MS = 1000


def summarize(path: Path) -> dict[str, Any]:
    report = json.loads((path / 'metrics.json').read_text())
    rows = report['evaluation_profile']
    keys = ('batch_wait_sec', 'within_batch_wait_sec', 'evaluation_wall_sec', 'evaluation_cpu_sec')
    timings = {key: dict(percentile([r[key]*MS for r in rows]),
        mean=float(np.mean([r[key]*MS for r in rows])), maximum=max(r[key]*MS for r in rows)) for key in keys}
    timings['non_cpu_upper_bound'] = percentile([
        max(0, r['evaluation_wall_sec']-r['evaluation_cpu_sec'])*MS for r in rows])
    stages = report.get('evaluation_stages', [])
    stage_keys = ('features', 'M0_G_fe', 'S1_S3', 'landing', 'legacy_features', 'counter', 'review', 'total_ms')
    stage_totals = {key: sum(r.get(key, 0.0) for r in stages) for key in stage_keys}
    gates = batch_gate_estimate(report) if not report.get('ipc') else []
    return dict(frames=report['frames'], latency_ms=report['latency_ms'],
        queue_max=report['evaluation_queue']['maximum'], timings_ms=timings,
        batch_gate_ms=percentile(gates), exclusive_stage_totals_ms=stage_totals,
        dropped=report['dropped_frames'], ipc=report.get('ipc'), feature_skips=report.get('feature_skips'))


def batch_gate_estimate(report: dict[str, Any]) -> list[float]:
    """旧threadの実バッチ開始と前バッチ終了から、固定待機中の重なりだけ復元する。"""
    rows, offset, previous_end, waits = report['evaluation_profile'], 0, 0.0, []
    for started, size in zip(report['batch_starts'], report['batch_sizes']):
        batch = rows[offset:offset+size]
        for row in batch:
            waits.append(max(0.0, started-max(row['recognized_at'], previous_end))*MS)
        if batch:
            previous_end = batch[-1]['finished_at']
        offset += size
    return waits


def differences(off: Path, on: Path) -> dict[str, Any]:
    left, right = [json.loads((path/'evaluations.json').read_text()) for path in (off, on)]
    if [r['frame'] for r in left] != [r['frame'] for r in right]:
        raise ValueError('ON/OFF比較は同じ全通知列で実行してください')
    absolute, signed, raw_absolute, raw_missing, source_changes = [], [], [], 0, 0
    for a, b in zip(left, right):
        source_changes += int(a['source'] != b['source'])
        signed.append(b['probability']-a['probability'])
        absolute.append(abs(signed[-1]))
        if a['raw_probability'] is not None and b['raw_probability'] is not None:
            raw_absolute.append(abs(b['raw_probability']-a['raw_probability']))
        else:
            raw_missing += 1
    return dict(frames=len(left), changed=sum(value != 0 for value in absolute),
        source_changes=source_changes, absolute=dict(percentile(absolute),
        mean=float(np.mean(absolute)), maximum=max(absolute)),
        raw_absolute=dict(percentile(raw_absolute),
            mean=float(np.mean(raw_absolute)) if raw_absolute else None,
            maximum=max(raw_absolute) if raw_absolute else None, excluded_frames=raw_missing),
        signed=dict(percentile(signed), mean=float(np.mean(signed)), minimum=min(signed), maximum=max(signed)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--profile', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('logs/live_b3_summary.json'))
    args = parser.parse_args()
    result = dict(equivalence=compare_arrays(args.reference/'display.npz', args.off/'display.npz'),
        coalesce_difference=differences(args.off, args.on),
        profiles={str(path): summarize(path) for path in args.profile})
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
