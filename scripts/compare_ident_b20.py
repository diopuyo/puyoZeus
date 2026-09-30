"""B20: 非realtime の OFF/ON 2実走について、認識通知列と評価recordsの同一性を全件比較する。

時刻・待ち深さ・壁時計など実行ごとに変わる列は除き、その列名と件数を明示する。
差があれば全件列挙する(数を出すだけにしない)。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

WALL_CLOCK_KEYS = frozenset({'captured_at', 'recognized_at', 'evaluated_at', 'queue_depth'})


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding='utf-8'))


def compare_rows(left: list[dict], right: list[dict]) -> dict:
    keys = sorted(set(left[0]) - WALL_CLOCK_KEYS) if left else []
    diffs = []
    for i, (a, b) in enumerate(zip(left, right)):
        bad = [k for k in keys if a.get(k) != b.get(k)]
        if bad:
            diffs.append(dict(index=i, frame=a.get('frame'), keys=bad))
    counts: dict[str, int] = {}
    for d in diffs:
        for k in d['keys']:
            counts[k] = counts.get(k, 0)+1
    return dict(left=len(left), right=len(right), compared_keys=keys, differing_key_counts=counts, excluded_keys=sorted(WALL_CLOCK_KEYS),
                length_equal=len(left) == len(right), differing_rows=len(diffs), differences=diffs[:50])


def compare_events(left: Path, right: Path) -> dict:
    a, b = left.read_text(encoding='utf-8').splitlines(), right.read_text(encoding='utf-8').splitlines()
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return dict(left=len(a), right=len(b), length_equal=len(a) == len(b), differing_lines=len(diffs),
                first_differences=diffs[:20])


def compare_npz(left: Path, right: Path) -> dict:
    out = {}
    with np.load(left, allow_pickle=True) as a, np.load(right, allow_pickle=True) as b:
        for name in sorted(set(a.files) & set(b.files)):
            x, y = a[name], b[name]
            wall = name in ('captured_at', 'recognized_at', 'recognition_ms', 'queue_wait_ms', 'cpu_ms',
                            'acquired_at', 'read_ms', 'queue_put_ms')
            if wall or x.shape != y.shape:
                out[name] = dict(skipped_wall_clock=wall, shape_left=list(x.shape), shape_right=list(y.shape))
                continue
            equal = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == 'f' else (x == y)
            out[name] = dict(mismatched=int((~equal).sum()), n=int(x.size))
    return out


def compare_files(left: Path, right: Path, names: tuple[str, ...]) -> dict:
    """決定的な出力ファイルはバイト一致で照合する。"""
    return {name: dict(left_bytes=(left/name).stat().st_size, right_bytes=(right/name).stat().st_size,
                       identical=(left/name).read_bytes() == (right/name).read_bytes()) for name in names}


def main() -> None:
    base = Path(sys.argv[1])
    left_name, right_name = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ('off', 'on')
    off, on = base/f'ident_{left_name}', base/f'ident_{right_name}'
    report = dict(evaluations=compare_rows(load_rows(off/'evaluations.json'), load_rows(on/'evaluations.json')),
                  events=compare_events(off/'events.jsonl', on/'events.jsonl'),
                  recognition_npz=compare_npz(off/'recognition.npz', on/'recognition.npz'),
                  display_npz=compare_npz(off/'display.npz', on/'display.npz'),
                  settled_npz=compare_npz(off/'settled.npz', on/'settled.npz'),
                  files=compare_files(off, on, ('events.diagnostics.json', 'events.hidden_death.json',
                                                'events.midchain.json', 'review_data.csv')))
    (base/f'ident_report_{left_name}_vs_{right_name}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('differences',)}
                      if k not in ('recognition_npz', 'display_npz', 'settled_npz', 'files') else v
                      for k, v in report.items()}, indent=1, ensure_ascii=False)[:6000])


if __name__ == '__main__':
    main()
