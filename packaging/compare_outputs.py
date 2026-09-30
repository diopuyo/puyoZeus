"""同一入力・同一区間で流した 2 つの出力ディレクトリを行単位で比べる (同値確認)。

    python packaging/compare_outputs.py <A のディレクトリ> <B のディレクトリ> [--out report.json]

比較対象 (在るものだけ。無いファイルは missing として報告し、成功扱いにしない):
  display.npz / settled.npz : 全キー・全行を厳密比較 (NaN 同士は等しい)。float 列は最大絶対差も出す
  events.jsonl              : 1 行 = 1 イベントを文字列で厳密比較
  inputs.jsonl.gz           : 同上 (Windows のパス区切り '\\' と '/' の差は別カウントで区別)
  review_data.csv           : セル単位。数値は厳密一致と最大絶対差、非数値は文字列一致
時刻 (壁時計) を含む metrics.json / evaluations.json は本質的に別物なので比較しない。
母数 (比較した行数・セル数) を必ず併記する。母数 0 は「測っていない」であり一致ではない。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

NPZ_FILES = ('display.npz', 'settled.npz')
JSONL_FILES = ('events.jsonl', 'inputs.jsonl.gz')
CSV_FILES = ('review_data.csv',)
FIRST_MISMATCHES = 5
IGNORED_KEYS = frozenset({'video_id'})  # 動画の内部 ID (パス由来) は入力差ではない


def compare_npz(a: Path, b: Path) -> dict[str, Any]:
    with np.load(a, allow_pickle=True) as left, np.load(b, allow_pickle=True) as right:
        columns = sorted((set(left.files) | set(right.files)) - IGNORED_KEYS)
        report: dict[str, Any] = dict(rows_a=len(left['t_sec']) if 't_sec' in left else None,
                                       rows_b=len(right['t_sec']) if 't_sec' in right else None,
                                       columns={}, mismatched_columns=[])
        for name in columns:
            if name not in left or name not in right:
                report['columns'][name] = 'missing_in_one_side'
                report['mismatched_columns'].append(name)
                continue
            report['columns'][name] = column_report(left[name], right[name])
            if report['columns'][name]['mismatched_rows']:
                report['mismatched_columns'].append(name)
    report['mismatched_rows_any_column'] = union_rows(report)
    return report


def column_report(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    count = min(len(x), len(y))
    x, y = x[:count], y[:count]
    if x.shape[1:] != y.shape[1:]:
        return dict(rows=count, mismatched_rows=count, note='shape_differs', rows_mask=list(range(count)))
    equal = x == y
    if x.dtype.kind == 'f':
        equal |= np.isnan(x) & np.isnan(y)
    flat = equal.reshape(count, -1).all(axis=1) if count else np.array([], dtype=bool)
    info: dict[str, Any] = dict(rows=count, length_diff=abs(len(x) - len(y)), mismatched_rows=int((~flat).sum()),
                                rows_mask=[int(i) for i in np.flatnonzero(~flat)[:1000]])
    if x.dtype.kind == 'f' and info['mismatched_rows']:
        with np.errstate(invalid='ignore'):
            info['max_abs_diff'] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
    return info


def union_rows(report: dict[str, Any]) -> int:
    rows: set[int] = set()
    for info in report['columns'].values():
        if isinstance(info, dict):
            rows |= set(info.get('rows_mask', []))
    return len(rows)


def read_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return stream.read().splitlines()


def compare_jsonl(a: Path, b: Path) -> dict[str, Any]:
    left, right = read_lines(a), read_lines(b)
    count = min(len(left), len(right))
    differing = [i for i in range(count) if left[i] != right[i]]
    separator_only = [i for i in differing if left[i].replace('\\\\', '/') == right[i].replace('\\\\', '/')]
    return dict(lines_a=len(left), lines_b=len(right), compared=count, mismatched=len(differing),
                mismatched_excluding_path_separator=len(differing) - len(separator_only),
                path_separator_only=len(separator_only), length_diff=abs(len(left) - len(right)),
                first_mismatches=differing[:FIRST_MISMATCHES])


def compare_csv(a: Path, b: Path) -> dict[str, Any]:
    with a.open(encoding='utf-8-sig', newline='') as f1, b.open(encoding='utf-8-sig', newline='') as f2:
        left, right = list(csv.reader(f1)), list(csv.reader(f2))
    header_equal = left[:1] == right[:1]
    cells = mismatched_cells = numeric_diff_cells = 0
    rows_bad: set[int] = set()
    max_abs = 0.0
    columns: dict[str, int] = {}
    for r in range(1, min(len(left), len(right))):
        for c, (x, y) in enumerate(zip(left[r], right[r])):
            cells += 1
            if x == y:
                continue
            mismatched_cells += 1
            rows_bad.add(r)
            columns[left[0][c]] = columns.get(left[0][c], 0) + 1
            try:
                max_abs = max(max_abs, abs(float(x) - float(y)))
                numeric_diff_cells += 1
            except ValueError:
                pass
    return dict(rows_a=len(left) - 1, rows_b=len(right) - 1, compared_rows=min(len(left), len(right)) - 1,
                compared_cells=cells, header_equal=header_equal, mismatched_cells=mismatched_cells,
                mismatched_rows=len(rows_bad), numeric_mismatched_cells=numeric_diff_cells,
                max_abs_diff_numeric=max_abs, mismatched_cells_by_column=dict(sorted(columns.items())),
                first_bad_rows=sorted(rows_bad)[:FIRST_MISMATCHES])


def compare_dirs(a: Path, b: Path) -> dict[str, Any]:
    report: dict[str, Any] = dict(a=str(a), b=str(b), files={}, missing=[])
    plan = [(NPZ_FILES, compare_npz), (JSONL_FILES, compare_jsonl), (CSV_FILES, compare_csv)]
    for names, function in plan:
        for name in names:
            if not (a / name).is_file() or not (b / name).is_file():
                report['missing'].append(name)
                continue
            report['files'][name] = function(a / name, b / name)
    return report


def total_mismatches(report: dict[str, Any]) -> int:
    """全ファイルの不一致件数の合計 (path 区切りだけの差は含めない)。missing があれば -1 (未確認)。"""
    if report['missing'] or not report['files']:
        return -1
    total = 0
    for name, info in report['files'].items():
        total += info.get('mismatched_rows_any_column', 0) + info.get('mismatched_excluding_path_separator', 0)
        total += info.get('mismatched_cells', 0) if name.endswith('.csv') else 0
        total += info.get('length_diff', 0)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--out', type=Path)
    options = parser.parse_args()
    report = compare_dirs(options.a, options.b)
    report['total_mismatches'] = total_mismatches(report)
    text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    if options.out:
        options.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
