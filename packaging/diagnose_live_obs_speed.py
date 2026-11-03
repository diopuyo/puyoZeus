"""500ms超の原票を工程へ帰属し、静的callbackと純粋なIPC待ちを区別する。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_live_obs_speed import combined, RATIO
from summarize_live_obs_time_gate import read, stats


def stages(root: Path, role: str) -> dict:
    result = defaultdict(lambda: defaultdict(float))
    for row in read(root/f'spool/frame_timing_{role}.jsonl'):
        for key, value in row['stages'].items():
            result[row['frame']][key] += value['wall_ms']
    return result


def diagnose(run: Path) -> dict:
    root = run/'output'
    rows = combined(root)
    parent, child = stages(root, 'evaluation'), stages(root, 'event_evaluation')
    outliers = [dict(row, parent_stages_ms=parent[row['frame']], child_stages_ms=child[row['frame']])
                for row in rows if row['total_low_spec_ms'] > 500]
    return dict(run=str(run), frames=len(rows), ratio=RATIO, outliers=outliers,
        static_callback_ms=stats([r.get('event_static_callback', 0) for r in parent.values()
                                 if 'event_static_callback' in r]),
        note='親のevent_static_callbackはevent_ipc_waitの内数。子event_updateは親callback待ちを含む。重複加算しない。')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(diagnose(args.run), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
