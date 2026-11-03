"""統合B査読の同区間比較を、最大通知の排他的内訳とともに保存する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.summarize_rt_integration_20261002 import rows, distribution

ROOT = Path('logs/rt_review_20261002/with_assets')
PREVIOUS_MAX_SEC = 2612.6
PREVIOUS_PEAK_WINDOW_SEC = 0.1
START_SEC, END_SEC = 2579., 2650.


def summarize(name: str) -> dict[str, Any]:
    folder = ROOT/name
    notices = [r for r in rows(folder/'audit', 'notifications-*.jsonl')
               if START_SEC <= r['t_sec'] < END_SEC]
    assert notices, f'{name}: 有効通知なし'
    metrics = json.loads((folder/'metrics.json').read_text())
    samples = rows(folder/'audit', 'state-*.jsonl')
    worst = max(notices, key=lambda r: r['milliseconds'])
    peak_window = [r for r in notices if abs(r['t_sec']-PREVIOUS_MAX_SEC) <= PREVIOUS_PEAK_WINDOW_SEC]
    peak = max(peak_window, key=lambda r: r['milliseconds']) if peak_window else None
    model_times = [sum(v for k, v in r['exclusive_ms'].items()
                       if k.startswith('model:G_fe:Hist')) for r in notices]
    runtime = rows(folder, 'runtime.jsonl')
    return dict(notification_ms=distribution([r['milliseconds'] for r in notices]),
                slowest=worst, previous_peak_window_max=peak,
                static_histgbc_ms=distribution(model_times),
                configuration=samples[-1]['configuration'],
                resource_gate_bytes=int((folder/'resource_gate.txt').read_text()),
                nice=samples[-1]['nice'], pools=samples[-1]['pools'],
                frames=metrics['frames'], dropped_frames=metrics['dropped_frames'],
                errors=max(r['evaluation_errors'] for r in runtime),
                recognition_summary=json.loads((folder/'next_recolor_pair_guard.json').read_text()),
                audit_retention=json.loads((folder/'events.audit_retention.json').read_text()))


def main() -> None:
    result = dict(interval_sec=[START_SEC, END_SEC],
                  previous_max=dict(t_sec=PREVIOUS_MAX_SEC, milliseconds=627.922975,
                                    comparison_window_sec=PREVIOUS_PEAK_WINDOW_SEC),
                  **{name: summarize(name) for name in ('production', 'static_lr')})
    path = Path('docs/verification/rt_review_latency_20261002.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in ('production', 'static_lr'):
        value = result[name]
        print(name, json.dumps({key: value[key] for key in
              ('notification_ms', 'slowest', 'previous_peak_window_max', 'static_histgbc_ms',
               'configuration', 'frames', 'dropped_frames', 'errors')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
