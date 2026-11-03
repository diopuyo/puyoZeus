"""保存原票は変更せず、v4.8入力時刻の同一性で計上を訂正する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterator
import numpy as np

BASE = Path(_public_dev_text('<work>/exch_review_render'))
VIDEOS = (29, 32, 34, 39, 51, 61, 65, 79, 92, 94, 80)
RATIO = 1.6473854230
LIMIT_MS = 500.


def rows(path: Path) -> Iterator[dict]:
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sources(video: int) -> list[Path]:
    paths = [BASE/f'v4_8/predictions_{video}.jsonl']
    if video != 80:
        paths.append(BASE/f'relabel_20261016/v4_8/predictions_{video}.jsonl')
    return paths


def stats(values: list[float]) -> dict:
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist()),
                rows=len(values), over_500=sum(v > LIMIT_MS for v in values))


def load_source(video: int) -> dict:
    by_game = defaultdict(list)
    for path in sources(video):
        for row in rows(path):
            if 'v_post' in row:
                by_game[row['game']].append(row)
    for values in by_game.values():
        values.sort(key=lambda r: r['input_sec'])
        assert len({r['input_sec'] for r in values}) == len(values)
    return {g: ([r['input_sec'] for r in values], values) for g, values in by_game.items()}


def mapped(video: int, path: Path) -> Iterator[dict]:
    source = load_source(video)
    seen, previous = set(), None
    for index, row in enumerate(rows(path), 1):
        stamps, values = source[row['game']]
        position = bisect_right(stamps, row['input_sec']) - 1
        assert position >= 0
        old = values[position]
        key = (video, row['game'], old['input_sec'])
        reused = max(0., old['low_spec_ms'] - old.get('inference_ms', 0.) * RATIO)
        assert reused == row['reused_physics_ms'], (video, index, key)
        own = row['measured_ms'] * RATIO
        assert abs(own + reused - row['low_spec_ms']) < 1e-9
        first = key not in seen
        detail = dict(video=video, line=index, game=row['game'], input_sec=row['input_sec'],
            source_input_sec=old['input_sec'], first_use=first, own_ms=own, physics_ms=reused,
            conservative_ms=row['low_spec_ms'], corrected_ms=own + (reused if first else 0.),
            states=row['current_states'], stages_low_spec_ms={k: v * RATIO for k, v in row['stage']['ms'].items()},
            v48_timing={k: v for k, v in old.items() if k.endswith('_ms')},
            adjacent_equal_cost=bool(previous and previous['game'] == row['game'] and previous['reused_physics_ms'] == reused))
        seen.add(key)
        previous = row
        yield detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    totals, corrected, own, exceed, remaining, manifests = [], [], [], [], [], []
    counts, dominant = Counter(), Counter()
    with (args.out/'accounting_rows.jsonl').open('x') as stream:
        for video in VIDEOS:
            path = BASE/f'v4_9e_capped_rep/predictions_{video}.jsonl'
            for source in [path, *sources(video)]:
                manifests.append(dict(path=str(source), sha256=sha(source)))
            for row in mapped(video, path):
                totals.append(row['conservative_ms']); corrected.append(row['corrected_ms']); own.append(row['own_ms'])
                counts['first' if row['first_use'] else 'reuse'] += 1
                if row['conservative_ms'] > LIMIT_MS:
                    exceed.append(row)
                if row['corrected_ms'] > LIMIT_MS:
                    remaining.append(row)
                    stages = dict(row['stages_low_spec_ms'], v48_physics=row['physics_ms'])
                    dominant[max(stages, key=stages.get)] += 1
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print('計上照合', video, len(totals), flush=True)
    result = dict(ratio=RATIO, units='low_spec_ms', percentile_method='numpy linear',
        identity=['video', 'game', 'source_input_sec'], conservative=stats(totals), corrected=stats(corrected),
        own=stats(own), source_uses=dict(counts), original_over_500=exceed, corrected_over_500=remaining,
        strict_classification=dict(Counter('first' if r['first_use'] else 'reuse' for r in exceed)),
        approximation_disagreements=[r for r in exceed if r['adjacent_equal_cost'] == r['first_use']],
        corrected_dominant=dict(dominant), manifests=manifests)
    with (args.out/'accounting.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('conservative', 'corrected', 'strict_classification', 'corrected_dominant')}), flush=True)


if __name__ == '__main__':
    main()
