"""実ライブの通知遅延・RSS・採用機能の実行証拠を集計する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

MIB = 1024**2
MILLISECONDS = 1000.0
START_SEC, END_SEC = 2579., 2650.


def rows(directory: Path, pattern: str) -> list[dict]:
    return [json.loads(line) for path in sorted(directory.glob(pattern))
            for line in path.read_text(encoding='utf-8').splitlines() if line]


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return dict(count=0, p50=None, p95=None, maximum=None)
    return dict(count=len(values), p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)), maximum=float(max(values)))


def summarize(directory: Path) -> dict:
    metrics = json.loads((directory/'metrics.json').read_text())
    notices = rows(directory/'audit', 'notifications-*.jsonl')
    samples = rows(directory/'audit', 'state-*.jsonl')
    notices = [row for row in notices if START_SEC <= row['t_sec'] < END_SEC]
    runtime = rows(directory, 'runtime.jsonl')
    config = json.loads((directory/'recognition_config.json').read_text())
    retention = {key: max(row['retained'].get(key, 0) for row in samples)
                 for key in {key for row in samples for key in row['retained']}}
    parent = [row['evaluation_wall_sec']*MILLISECONDS for row in metrics['evaluation_profile']]
    return dict(worker_notification_ms=distribution([r['milliseconds'] for r in notices]),
        parent_notification_ms=distribution(parent),
        worker_rss_mib=distribution([r['rss_bytes']/MIB for r in samples]),
        worker_rss_first_last_mib=[samples[i]['rss_bytes']/MIB for i in (0, -1)],
        configuration=samples[-1]['configuration'], model_calls=samples[-1]['model_calls'],
        recognition_config=config, retained_maxima=retention,
        history_games=[r['history_games'] for r in samples if r['history_games']],
        resource_gate_bytes=int((directory/'resource_gate.txt').read_text()),
        nice=samples[-1]['nice'], pools=samples[-1]['pools'],
        frames=metrics['frames'], dropped_frames=metrics['dropped_frames'],
        game_starts=runtime[-1]['game_starts'],
        evaluation_errors=max(r['evaluation_errors'] for r in runtime),
        processing_error_records=len(rows(directory, 'evaluation_errors.jsonl')),
        evaluation_restarts=max(r['evaluation_restarts'] for r in runtime),
        assets=metrics['assets'], loadavg=[metrics['loadavg_start'], metrics['loadavg_end']])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('logs/rt_integration_20261002/video_runs'))
    root = parser.parse_args().output
    result = {name: summarize(root/name) for name in ('baseline', 'integrated')}
    result['interval_sec'] = [START_SEC, END_SEC]
    result['baseline_commit'] = '8fe251c'
    result['baseline_connection_repair'] = ['src/phase_j/live_bridge.py', 'src/phase_j/live_process.py']
    result['unmodified_baseline'] = 'placement_reconcile logging AttributeError before recognition'
    (root/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ('baseline', 'integrated'):
        print(name, json.dumps({key: value for key, value in result[name].items()
              if key not in ('history_games', 'recognition_config')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
