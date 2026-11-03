"""全通知の判定データとCNN生出力をバイト照合する。時刻を丸めない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def board_counts(before: list, after: list) -> dict[str, Any]:
    """盤面があるセルの母数と欠測の不一致を、種類別に数える。"""
    fields = ('raw_cnn_board', 'cnn_board', 'confirmed_board')
    counts = {name: dict(cells=0, differences=0, missing_mismatch=0) for name in fields}
    stable = 0
    for left, right in zip(before, after):
        a, b = pickle.loads(left.result_bytes), pickle.loads(right.result_bytes)
        stable += a.p1.state.name == a.p2.state.name == 'STABLE'
        for side in ('p1', 'p2'):
            for name in fields:
                x, y = getattr(getattr(a, side), name), getattr(getattr(b, side), name)
                if x is None or y is None:
                    counts[name]['missing_mismatch'] += (x is None) != (y is None)
                else:
                    counts[name]['cells'] += x._grid.size
                    counts[name]['differences'] += int(np.count_nonzero(x._grid != y._grid))
    return dict(boards=counts, both_stable_frames=stable)


def compare(base: Path, candidate: Path) -> dict[str, Any]:
    """壁時計captured_at/recognized_atを除き、映像時刻と全判定を比較する。"""
    with (base/'notices.pickle').open('rb') as stream:
        before = pickle.load(stream)
    with (candidate/'notices.pickle').open('rb') as stream:
        after = pickle.load(stream)
    mismatches = []
    for index, (left, right) in enumerate(zip(before, after)):
        fields = ('frame', 't_sec', 'result_bytes', 'pipeline', 'dropped_before', 'b3')
        for field in fields:
            if pickle.dumps(getattr(left, field)) != pickle.dumps(getattr(right, field)):
                mismatches.append([index, field])
    keys_equal = (base/'keys.json').read_bytes() == (candidate/'keys.json').read_bytes()
    logits_bad, argmax_bad, max_diff, patches = 0, 0, 0.0, 0
    with np.load(base/'logits.npz') as left, np.load(candidate/'logits.npz') as right:
        if keys_equal:
            for key in left.files:
                a, b = left[key], right[key]
                logits_bad += a.tobytes() != b.tobytes()
                patches += len(a)
                argmax_bad += int(np.count_nonzero(a.argmax(axis=1) != b.argmax(axis=1)))
                max_diff = max(max_diff, float(np.max(np.abs(a-b))))
        calls = len(left.files)
    return dict(base=base.name, candidate=candidate.name, notices_before=len(before),
        notices_after=len(after), mismatches=mismatches, logits_input_keys_equal=keys_equal,
        logits_calls=calls, patches=patches, logits_mismatch_calls=logits_bad if keys_equal else None,
        argmax_mismatch=argmax_bad if keys_equal else None,
        logits_max_abs_diff=max_diff if keys_equal else None, **board_counts(before, after))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    parser.add_argument('candidates', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [compare(args.base, path) for path in args.candidates]
    args.out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for row in rows:
        print(row['candidate'], '通知差', len(row['mismatches']),
              'logits差', row['logits_mismatch_calls'], '最大差', row['logits_max_abs_diff'])


if __name__ == '__main__':
    main()
