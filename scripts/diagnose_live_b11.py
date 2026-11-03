"""B10保存ログの段別時間と、検収区間末尾の実試合を監査する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.analyze_live_b9 import lines
from scripts.measure_live_b6 import save
from scripts.measure_live_b9 import START_SEC, END_SEC
from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO

WINDOW_SEC = 300.0
LAST_START_SEC = 3412.2166666666667
LAST_PLAYING_SEC = 3416.0


def stage_windows(path: Path) -> list[dict[str, Any]]:
    metrics = json.loads((path/'metrics.json').read_text())
    output = []
    with np.load(path/'recognition.npz') as recognition:
        for lo in np.arange(START_SEC, END_SEC, WINDOW_SEC):
            hi = min(lo+WINDOW_SEC, END_SEC)
            mask = (recognition['t_sec'] >= lo) & (recognition['t_sec'] < hi)
            row: dict[str, Any] = dict(start=float(lo), end=float(hi))
            for key in ('recognition_ms', 'cpu_ms', 'queue_put_ms'):
                values = recognition[key][mask]
                row[key] = dict(count=len(values), p50=float(np.percentile(values, 50)),
                                p95=float(np.percentile(values, 95)))
            for key in ('state_updates', 'probability_calculations'):
                values = [r['milliseconds'] for r in metrics[key] if lo <= r['t_sec'] < hi]
                row[key] = dict(count=len(values), p50=float(np.percentile(values, 50)),
                                p95=float(np.percentile(values, 95)))
            output.append(row)
    return output


def boundary_evidence(path: Path, output: Path, video: Path) -> dict[str, Any]:
    starts = lines(path/'runtime.jsonl')[-1]['game_starts']
    capture = cv2.VideoCapture(str(video))
    try:
        for name, t_sec in (('extra_start', LAST_START_SEC), ('extra_boundary', LAST_PLAYING_SEC)):
            capture.set(cv2.CAP_PROP_POS_MSEC, t_sec*1000)
            ok, frame = capture.read()
            if not ok or not cv2.imwrite(str(output/(name+'.png')), frame):
                raise RuntimeError('境界証拠画像を保存できません')
    finally:
        capture.release()
    return dict(detected_starts=len(starts), detected_internal_boundaries=len(starts)-1,
        final_start=starts[-1], first_match=41, last_match=58,
        explanation='末尾は勝数28対29から始まる第58試合。得点リセットは実際の次試合であり誤検出ではない。',
        original_interval=[START_SEC, END_SEC], evidence=['extra_start.png', 'extra_boundary.png'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('logs/live_b9/long'))
    parser.add_argument('--output', type=Path, default=Path('logs/live_b11'))
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    save(options.output/'b10_diagnosis.json', dict(stages=stage_windows(options.source),
        boundary=boundary_evidence(options.source, options.output, options.video)))


if __name__ == '__main__':
    main()
