"""55分試験の段別時刻を全件集計し、短報の未測定欄を更新する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/live_parallel'
MS = 1000


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), maximum=max(values))


def summarize(mode: str) -> dict[str, Any]:
    """認識wallとcapture→親評価finishを混ぜず、全体CPUは別に記録する。"""
    run = OUT/f'camera_{mode}'
    recognition, end_to_end, captured = [], [], []
    for path in sorted((run/'output/spool').glob('frame_timing_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['role'] == 'recognition':
                    recognition.append(float(row['wall_ms']))
                    captured.append(float(row['captured_at']))
                elif row['role'] == 'evaluation':
                    end_to_end.append((row['finished_at']-row['captured_at'])*MS)
    if not recognition or not end_to_end:
        raise RuntimeError(f'{mode}: 認識または親評価の計装が0件')
    captured.sort()
    common = json.loads((run/'SOAK_RESULT.json').read_text(encoding='utf-8'))
    return dict(mode=mode, recognition_ms=stats(recognition),
        capture_to_evaluation_finish_ms=stats(end_to_end),
        capture_fps=(len(captured)-1)/(captured[-1]-captured[0]),
        cpu=common['cpu'], memory=common['memory'], status=common['status'])


def main() -> None:
    rows = [summarize(mode) for mode in ('off', 'on')]
    (OUT/'camera_comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    report = Path(_public_dev_text('<work>/LIVE_PAR.md'))
    lines = report.read_text(encoding='utf-8').splitlines()
    def latency(row: dict) -> str:
        value = row['capture_to_evaluation_finish_ms']
        return '/'.join(f'{value[key]:.1f}' for key in ('p50', 'p95', 'p99', 'maximum'))
    for index, line in enumerate(lines):
        if line.startswith('55分試験:'):
            lines[index] = ('55分試験: 内部m→候補の計装集計済み。fps '
                f"{rows[0]['capture_fps']:.2f}→{rows[1]['capture_fps']:.2f}。完走/終了状態はcamera_comparison.json参照。")
        elif line.startswith('capture→評価終了:'):
            lines[index] = f'capture→評価終了: P50/P95/P99/最大msはm {latency(rows[0])} → 候補 {latency(rows[1])}。'
    assert len(lines) <= 25
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
