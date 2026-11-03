"""固定画素比較の母数・最初の差・認識段階をまとめる。許容差は導入しない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_outputs import compare_dirs, total_mismatches

RECOGNITION_KEYS = ('frame', 't_sec', 'boards', 'raw', 'stable', 'states', 'active', 'ready', 'scores')
B3_VALUE_KEYS = ('frame', 't_sec', 'trigger', 'raw_probability', 'error')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def compare_b3(left: Path, right: Path) -> dict:
    """計算時間・壁時計を除き、B3の全判定値と失敗状態を厳密比較する。"""
    values = []
    for directory in (left, right):
        path = directory / 'b3_updates.jsonl'
        if not path.is_file():
            return dict(available=False)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        values.append([{key: row[key] for key in B3_VALUE_KEYS} for row in rows])
    a, b = values
    return dict(available=bool(a and b), rows=[len(a), len(b)], identical=a == b,
                errors=[sum(row['error'] is not None for row in rows) for rows in values])


def pixel_rows(directory: Path) -> list[dict]:
    """認識に実際に渡した全フレームを取り出す。メタデータだけのprobeは除外。"""
    candidates = [read_json(path) for path in directory.glob('pixels-*.json')]
    if not candidates:
        raise ValueError(f'入力画素監査がありません: {directory}')
    return max(candidates, key=len)


def first_array_difference(left: Path, right: Path, keys: tuple[str, ...] | None = None) -> dict | None:
    """列名順でなく時刻順の最初の差を、値付きで返す。"""
    with np.load(left, allow_pickle=True) as a, np.load(right, allow_pickle=True) as b:
        columns = keys or tuple(sorted(set(a.files) & set(b.files) - {'video_id'}))
        found = []
        for key in columns:
            x, y = a[key], b[key]
            count = min(len(x), len(y))
            eq = x[:count] == y[:count]
            if x.dtype.kind == 'f':
                eq |= np.isnan(x[:count]) & np.isnan(y[:count])
            bad = np.flatnonzero(~eq.reshape(count, -1).all(axis=1))
            if len(bad):
                row = int(bad[0])
                found.append(dict(row=row, column=key, t_sec=float(a['t_sec'][row]),
                                  left=x[row].tolist(), right=y[row].tolist()))
        return min(found, key=lambda row: row['row']) if found else None


def summarize_pair(left: Path, right: Path, target: Path) -> dict:
    """既存の全出力比較に、同一入力・設定・分解記録の証拠を併記する。"""
    report = compare_dirs(left, right)
    report['total_mismatches'] = total_mismatches(report)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pixels_a, pixels_b = pixel_rows(left), pixel_rows(right)
    execution_a, execution_b = (read_json(path / 'execution.json') for path in (left, right))
    summary = dict(left=str(left), right=str(right), complete=all((p/'DONE').is_file() for p in (left, right)),
        pixels=dict(left=len(pixels_a), right=len(pixels_b), identical=pixels_a == pixels_b),
        recognition_config_equal=read_json(left/'recognition_config.json') == read_json(right/'recognition_config.json'),
        settings_equal={key: execution_a[key] == execution_b[key]
                        for key in ('seed', 'production_flags', 'manifest_sha256')},
        backends=[execution_a['backend'], execution_b['backend']],
        frames=[read_json(path/'metrics.json')['frames'] for path in (left, right)],
        dropped=[read_json(path/'metrics.json')['dropped_frames'] for path in (left, right)],
        missing=report['missing'], total_mismatches=report['total_mismatches'], files={})
    for name, info in report['files'].items():
        summary['files'][name] = {key: value for key, value in info.items()
            if key not in ('columns', 'mismatched_cells_by_column', 'first_bad_rows', 'first_mismatches')}
    summary['first_display_difference'] = first_array_difference(left/'display.npz', right/'display.npz')
    audit = 'recognition.npz'
    if (left/audit).is_file() and (right/audit).is_file():
        summary['first_recognition_difference'] = first_array_difference(left/audit, right/audit, RECOGNITION_KEYS)
    summary['b3'] = compare_b3(left, right)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = summarize_pair(args.left, args.right, args.output.with_suffix('.full.json'))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
