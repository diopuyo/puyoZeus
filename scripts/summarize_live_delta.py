"""差分認識の原票を集計する（未測定の表示品質は推測しない）。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
from pathlib import Path

import numpy as np

ROOT = Path(_public_dev_text('<work>/delta'))
WARMUP = 60


def compare(reference: list[dict], candidate: list[dict]) -> dict:
    assert len(reference) == len(candidate)
    counts = dict(frames=len(reference), raw_cells=0, raw_diff=0, stable_cells=0,
                  stable_diff=0, confirmed_frames=0, confirmed_diff=0, digest_diff=0,
                  both_stable_cells=0, both_stable_diff=0, both_stable_frames=0)
    for before, after in zip(reference, candidate):
        counts['digest_diff'] += before['digest'] != after['digest']
        raw = np.asarray(before['raw'])
        counts['raw_cells'] += raw.size
        counts['raw_diff'] += int(np.count_nonzero(raw != np.asarray(after['raw'])))
        if before['states'] == after['states'] == ['STABLE', 'STABLE']:
            if None not in before['confirmed'] and None not in after['confirmed']:
                left, right = np.asarray(before['confirmed']), np.asarray(after['confirmed'])
                counts['both_stable_frames'] += 1
                counts['both_stable_cells'] += left.size
                counts['both_stable_diff'] += int(np.count_nonzero(left != right))
        if before['confirmed'] != [None, None] or after['confirmed'] != [None, None]:
            counts['confirmed_frames'] += 1
            counts['confirmed_diff'] += before['confirmed'] != after['confirmed']
        for side in range(2):
            if before['states'][side] == after['states'][side] == 'STABLE':
                left, right = before['confirmed'][side], after['confirmed'][side]
                if left is not None and right is not None:
                    counts['stable_cells'] += np.asarray(left).size
                    counts['stable_diff'] += int(np.count_nonzero(np.asarray(left) != np.asarray(right)))
    return counts


def main() -> None:
    baseline = json.loads((ROOT / 'baseline.json').read_text())
    summary = {}
    for name in ('baseline', 'off1', 'dedup', 'delta1', 'off2', 'delta2'):
        path = ROOT / (name + '.json')
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        values = np.asarray([r['ms'] for r in rows[WARMUP:]])
        summary[name] = dict(compare(baseline, rows), timing_n=len(values),
            mean_ms=float(values.mean()), p95_ms=float(np.percentile(values, 95)),
            pipeline_fps=float(1000 / values.mean()),
            receipt=json.loads(path.with_suffix('.receipt.json').read_text()))
    (ROOT / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
