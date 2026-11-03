"""速度原票の同居負荷・CPU固定・遅い工程を小さい検収票へまとめる。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_live_obs_speed import combined
from summarize_live_obs_time_gate import read, stats


def audit(run: Path) -> dict:
    root = run/'output'
    rows = combined(root)
    external = read(run/'external_load.jsonl')
    resources = read(root/'b3_resources.jsonl')
    processes = [p for r in resources for p in r['processes'] if p['name'].lower() != 'obs64.exe']
    others = {p['pid']: p for r in external for p in r['processes'] if p['name'].lower() != 'obs64.exe'}
    times = sorted(set(r['t_sec'] for r in rows))
    outliers = [r for r in rows if r['total_low_spec_ms'] > 500]
    top = sorted(outliers, key=lambda r: r['total_low_spec_ms'], reverse=True)[:10]
    result = dict(run=str(run), external_samples=len(external), external_python=list(others.values()),
        affinity_masks=sorted(set(p['affinity_mask'] for p in processes)),
        priority_classes=sorted(set(p['priority_class'] for p in processes)),
        observation_interval_ms=stats([(b-a)*1000 for a, b in zip(times, times[1:])]),
        common_599_720_total_ms=stats([r['total_low_spec_ms'] for r in rows if r['t_sec'] < 720]),
        outliers=len(outliers), dominant_outlier_stages=dict(Counter(
            max(r['measured_ms'], key=r['measured_ms'].get) for r in outliers)),
        top_outliers=[dict(frame=r['frame'], t_sec=r['t_sec'], total_low_spec_ms=r['total_low_spec_ms'],
                           measured_ms=r['measured_ms']) for r in top])
    result['slow_publications'] = [dict(frame=r['frame'], wall_ms=r['wall_ms'], cpu_ms=r['cpu_ms'])
        for r in sorted(read(root/'spool/frame_timing_publication.jsonl'),
                        key=lambda r: r['wall_ms'], reverse=True)[:5]]
    return result


if __name__ == '__main__':
    value = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k not in ('top_outliers',)}, ensure_ascii=False))
