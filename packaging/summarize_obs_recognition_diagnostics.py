"""認識内部と初回importの原因票。並行・入れ子の時間は合算しない。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_live_obs_time_gate import read, stats


def summarize(directory: Path) -> dict:
    result = {}
    for role in ('recognition', 'evaluation', 'event_evaluation'):
        rows = read(directory/('frame_timing_'+role+'.jsonl'))
        stages: dict = defaultdict(list)
        for row in rows:
            for name, values in row['stages'].items():
                if values['calls']:
                    stages[name].append(values)
        result[role] = dict(rows=len(rows), stages={name:dict(
            wall_ms=stats([v['wall_ms'] for v in values]),
            thread_cpu_ms=stats([v['cpu_ms'] for v in values]),
            process_cpu_ms=stats([v['process_cpu_ms'] for v in values if 'process_cpu_ms' in v]))
            for name, values in stages.items()},
            slowest_rows=sorted(rows, key=lambda r: r['wall_ms'], reverse=True)[:5],
            slow_imports=sorted([dict(frame=r['frame'], t_sec=r['t_sec'], **item)
                for r in rows for item in r['details'].get('slow_imports', ())],
                key=lambda r: r['wall_ms'], reverse=True)[:20])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.directory), ensure_ascii=False, indent=2), encoding='utf-8')
