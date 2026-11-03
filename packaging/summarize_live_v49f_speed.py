"""同一frameに結合し、更新の処理量と実際の公開遅延を分けて集計する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT = Path(_public_dev_text('<work>/build_20261019a_stage/verification_r2'))
OUT = Path('D:/puyo_analyzer/packaging/build_20261019a_release')
RATIO = 1.6473854230
LIMIT = 500.


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding='utf-8-sig')]


def stats(values: list[float]) -> dict:
    return dict(n=len(values), p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)), maximum=max(values), over_500=sum(x > LIMIT for x in values))


def intervals_ms(values: list[tuple]) -> float:
    end, cost = float('-inf'), 0.
    for start, finish in sorted(values):
        cost += max(0., finish-max(start, end))
        end = max(end, finish)
    return cost*1000.


def collect() -> tuple[list[dict], dict, list]:
    data, intervals = ROOT/'output', defaultdict(list)
    recognition = {r['frame']: r for r in rows(data/'spool/frame_timing_recognition.jsonl') if r['frame'] is not None}
    b3, v49 = defaultdict(float), defaultdict(float)
    for row in rows(data/'b3_state_timing.jsonl'):
        b3[row['frame']] += row.get('state_ms', 0)+row.get('ledger_ms', 0)
    for row in rows(data/'b3_updates.jsonl'):
        b3[row['frame']] += sum(row.get(k, 0) for k in ('search_ms', 'feature_only_ms', 'inference_ms'))
    values = rows(data/'v49f_updates.jsonl')
    for row in values:
        v49[row['id']] += row['compute_ms']
    for role in ('evaluation', 'publication'):
        for row in rows(data/f'spool/frame_timing_{role}.jsonl'):
            intervals[row['frame']].append((row['started_at'], row['finished_at']))
    combined = []
    updates = {r['id'] for r in values if r['value'] is not None}
    for frame, row in sorted(recognition.items()):
        parts = dict(recognition=row['wall_ms'], b3=b3[frame], v49f=v49[frame],
                     display_dispatch=intervals_ms(intervals[frame]))
        combined.append(dict(frame=frame, t=row['t_sec'], v49f_updated=frame in updates,
            measured_ms=parts, low_spec_ms={k: v*RATIO for k, v in parts.items()}, total=sum(parts.values())*RATIO))
    return combined, recognition, values


def main() -> None:
    combined, recognition, values = collect()
    published, seen = [], set()
    valid_polls, examples = 0, []
    by_completion = {r['completed_at']: r for r in values}
    for row in rows(ROOT/'polls.jsonl'):
        value = row['snapshot'].get('evaluations', {}).get('v49f', {})
        if value.get('p1') is None:
            continue
        valid_polls += 1
        stamp = value.get('completed_at')
        if stamp in seen or stamp not in by_completion:
            continue
        seen.add(stamp)
        source = recognition.get(by_completion[stamp]['id'])
        if source:
            published.append((row['at']-source['captured_at'])*1000*RATIO)
        if len(examples) < 3:
            examples.append(value)
    result = dict(ratio=RATIO, limit_ms=LIMIT, publication_poll_interval_ms=100,
        valid_v49f_polls=valid_polls, examples=examples, v49f_notifications=len(values),
        v49f_updates=sum(r['value'] is not None for r in values),
        fifo_wait_measured_ms=stats([r['wait_ms'] for r in values]),
        v49f_compute_measured_ms=stats([r['compute_ms'] for r in values]),
        first_publication_low_spec_ms=stats(published), browser_paint_measured=False)
    for name, selected in [('all_frames', combined), ('v49f_update_frames', [r for r in combined if r['v49f_updated']])]:
        result[name] = {key: stats([r['low_spec_ms'][key] for r in selected]) for key in selected[0]['low_spec_ms']}
        result[name]['total'] = stats([r['total'] for r in selected])
    (OUT/'speed_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined), encoding='utf-8')
    (OUT/'speed_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
