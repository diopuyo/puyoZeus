"""B4計測の母数付き集計と、混同・位置・時刻の監査CSVを保存する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_live_pipeline_20260928 import save_json
from scripts.verify_live_b4_degradation import START, FPS, COLORS

LOGS = Path('logs')
DESTINATION = Path('docs/agent_coordination')
PREFIX = 'PHASE_J_B4_2026-09-28'
LIVE = 'live_b4_realtime_verified'
COLOR_NAMES = ('空', '赤', '青', '緑', '黄', '紫', 'おじゃま', '不明')


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def live_summary(name: str) -> dict:
    report = read(LOGS / name / 'metrics.json')
    selected = ('frames', 'expected_frames', 'dropped_frames_in_measured_window',
                'latency_ms', 'ipc', 'counter_search', 'nice', 'gpu', 'coalesce_features',
                'loadavg_start', 'loadavg_end', 'assets')
    result = {key: report[key] for key in selected}
    result['queue_maximum'] = report['evaluation_queue']['maximum']
    result['queue_end'] = report['evaluation_queue']['end']
    result['evaluation_mean_ms'] = {
        key: float(np.mean([r[key] for r in report['evaluation_profile']])*1000)
        for key in ('evaluation_wall_sec', 'evaluation_cpu_sec')}
    rows = read(LOGS / name / 'evaluations.json')
    result['pending_frames'] = sum(r.get('counter_search', {}).get('pending', False) for r in rows)
    result['counter_dispatch_max_ms'] = max(r['counter'] for r in report['evaluation_stages'])
    return result


def csv_reports(report: dict, crf: int) -> None:
    """基準→変換後の全色混同と、差分セルの対応時刻・位置を永続化する。"""
    detail = report['baseline']['breakdown']
    with (DESTINATION / f'{PREFIX}_CRF{crf}_confusion.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['元動画/変換後（相違セルのみ）', *COLOR_NAMES])
        for name, row in zip(COLOR_NAMES, detail['confusion']):
            writer.writerow([name, *row])
    folder = LOGS / 'live_b4_degradation'
    with np.load(folder / 'original.npz') as left, np.load(folder / f'crf{crf}_baseline.npz') as right:
        with (DESTINATION / f'{PREFIX}_CRF{crf}_cells.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['t_sec', 'side', 'row', 'col', '元色', '変換後色',
                             '元1P状態', '元2P状態', '変換1P状態', '変換2P状態', '瞬間観測も相違'])
            for frame, side, row, col in detail['indices']:
                index = (frame, side, row, col)
                writer.writerow([START+frame/FPS, side+1, row, col,
                    int(left['boards'][index]), int(right['boards'][index]),
                    *left['states'][frame], *right['states'][frame],
                    bool(left['raw'][index] != right['raw'][index])])


def strip_indices(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_indices(item) for key, item in value.items() if key != 'indices'}
    if isinstance(value, list):
        return [strip_indices(item) for item in value]
    return value


def main() -> None:
    baseline = read(LOGS / 'live_b4_baseline/gpu/breakdown.json')
    report = dict(live=live_summary(LIVE),
        baseline_conditions=read(LOGS / 'live_b4_baseline/conditions.json'),
        gpu_baseline={key: baseline[key] for key in ('frames', 'decoded_frames', 'processing_seconds',
            'processing_fps', 'stages_ms', 'gpu', 'nice', 'parallel_count', 'no_render')},
        degradation={})
    for crf in (23, 35):
        quality = read(LOGS / f'live_b4_degradation/crf{crf}.json')
        csv_reports(quality, crf)
        report['degradation'][str(crf)] = strip_indices(quality)
    save_json(DESTINATION / f'{PREFIX}.json', report)
    print(json.dumps(dict(latency=report['live']['latency_ms']['capture_to_sse'],
        degradation={key: {mode: {name: value[mode][name] for name in
            ('different_cells', 'cells', 'stable_availability_mismatches')}
            for mode in ('baseline', 'calibrated')}
            for key, value in report['degradation'].items()}), indent=2))


if __name__ == '__main__':
    main()
