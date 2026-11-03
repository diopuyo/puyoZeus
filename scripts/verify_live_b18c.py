"""B18bの全一致を維持したまま、同一認識観測で窓処理の前後を測る。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np

from scripts import verify_live_b18b as previous
from scripts.measure_live_b18c import paired_overlay
from scripts.profile_live_b18c import OUT
from scripts.summarize_live_b18b import compare_acceptance, compare_windows, complete_display_trace
from scripts.verify_live_b18a import compare_rows, quantiles
from scripts.run_live_pipeline_20260928 import compare_arrays


def offline_compare(record: Path, dest: Path) -> None:
    expected = previous.replay(record, dest/'offline', False)
    actual = json.loads((dest.parent/'probabilities.json').read_text())
    previous.save(dest/'comparison.json', dict(probabilities=compare_rows(expected, actual),
        display=compare_arrays(dest/'offline/display.npz', dest.parent/'display.npz')))


def same_record(before: Path, after: Path) -> dict:
    """B18bが保存した評価境界の入力を、型情報も含めて全行比較する。"""
    def updates(path: Path) -> Any:
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'update':
                    yield row
    equal, total, first, fields = 0, 0, None, Counter()
    for index, (old, new) in enumerate(zip_longest(updates(before), updates(after))):
        total += 1
        equal += old == new
        if old != new and first is None:
            first = dict(index=index, offline=old, realtime=new)
        if old != new:
            fields.update(different_fields(old, new))
    return dict(equal=equal, total=total, first_mismatch=first, different_fields=dict(fields))


def different_fields(before: Any, after: Any, path: str = '') -> list[str]:
    """過去原票の入力差を列単位で保存し、現行コミットとの同時計測と区別する。"""
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [field for key in sorted(before.keys() | after.keys())
                for field in different_fields(before.get(key), after.get(key), path+'.'+key)]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [field for idx, (a, b) in enumerate(zip(before, after))
                for field in different_fields(a, b, path+f'.{idx}')]
    return [path]


def report(source: str, dest: Path, instances: list | None = None) -> None:
    old = Path('logs/live_b18b')/('video/review' if source == 'review' else 'video-final/zenchi')
    with np.load(dest/'timing.npz') as timings:
        costs = {key: quantiles(timings[key]) for key in timings.files}
        costs['before_retention_sec'] = quantiles(timings['before_snapshot_sec']-timings['before_hsv_sec'])
    paired = (json.loads((dest/'report.json').read_text())['paired_observation'] if instances is None else
        dict(total=sum(v.paired_rows for v in instances), equal=sum(v.paired_equal for v in instances),
             first_mismatch=next((v.first_difference for v in instances if v.first_difference), None)))
    result = dict(source=source, windows=compare_windows(source, dest),
        acceptance=compare_acceptance(source, dest), cost=costs,
        same_input_replay=json.loads((dest/'comparison.json').read_text()),
        b18b_inputs=same_record(old/'inputs.jsonl.gz', dest/'inputs.jsonl.gz'),
        b18b_probabilities=compare_rows(complete_display_trace(old/'probabilities.json'),
                                      complete_display_trace(dest/'probabilities.json')),
        b18b_display=compare_arrays(old/'display.npz', dest/'display.npz'),
        paired_observation=paired)
    previous.save(dest/'report.json', result)
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'mismatches'}
        if isinstance(value, dict) else value for key, value in result.items()}), flush=True)


def video(source: str, dest: Path) -> None:
    from src.phase_j import live_snapshot
    paired, instances = paired_overlay(), []
    def create(reader: Any) -> Any:
        value = paired(reader)
        instances.append(value)
        return value
    with ExitStack() as stack:
        stack.enter_context(patch.object(live_snapshot, 'LiveSnapshotInputs', create))
        stack.enter_context(patch.object(previous, 'compare', offline_compare))
        previous.capture(source, dest)
    report(source, dest, instances)


def main() -> None:
    from src.phase_j.live_cpu import configure_environment, apply_runtime
    configure_environment(1, 10)
    apply_runtime('recognition')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('review', 'zenchi'), required=True)
    parser.add_argument('--saved', action='store_true')
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        report(args.source, args.output_root/'video'/args.source)
    elif args.saved:
        previous.compare(Path('logs/e31/records')/f'{args.source}.jsonl.gz', args.output_root/'saved'/args.source)
    else:
        video(args.source, args.output_root/'video'/args.source)


if __name__ == '__main__':
    main()
