"""packaging/compare_outputs.py (同値確認の比較器) の試験。

比較器が壊れていれば「差 0」は無意味になる (測定器事故の前例)。陽性対照 (差を入れれば必ず検出する) を先に固定する。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))

import compare_outputs as co  # noqa: E402

ROWS = 50


def make_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    t = np.arange(ROWS, dtype=np.float64) / 30
    np.savez(root / 'display.npz', t_sec=t, prob=np.linspace(0, 1, ROWS), grids=np.zeros((ROWS, 13, 6), np.int8),
             video_id=np.array('x'))
    np.savez(root / 'settled.npz', t_sec=t, grids=np.ones((ROWS, 13, 6), np.int8))
    (root / 'events.jsonl').write_text('\n'.join(json.dumps(dict(i=i)) for i in range(ROWS)), encoding='utf-8')
    with gzip.open(root / 'inputs.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.write('\n'.join(json.dumps(dict(i=i, model_dir='models\\v3')) for i in range(ROWS)))
    (root / 'review_data.csv').write_text('a,b\n' + '\n'.join(f'{i},{i * 0.5}' for i in range(ROWS)), encoding='utf-8')
    return root


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    a = make_dir(tmp_path / 'a')
    shutil.copytree(a, tmp_path / 'b')
    return a, tmp_path / 'b'


def test_identical_dirs_report_zero_with_population(pair: tuple[Path, Path]) -> None:
    report = co.compare_dirs(*pair)
    assert co.total_mismatches({**report}) == 0
    assert report['files']['review_data.csv']['compared_cells'] == ROWS * 2
    assert report['files']['events.jsonl']['compared'] == ROWS and report['files']['display.npz']['rows_a'] == ROWS


def test_positive_control_npz_float_change_is_detected(pair: tuple[Path, Path]) -> None:
    a, b = pair
    with np.load(b / 'display.npz') as data:
        arrays = dict(data)
    arrays['prob'][7] += 1e-12  # 最小級の差でも検出する
    np.savez(b / 'display.npz', **arrays)
    report = co.compare_dirs(a, b)
    info = report['files']['display.npz']
    assert info['mismatched_columns'] == ['prob'] and info['mismatched_rows_any_column'] == 1
    assert co.total_mismatches(report) == 1 and info['columns']['prob']['max_abs_diff'] > 0


def test_positive_control_board_cell_change_is_detected(pair: tuple[Path, Path]) -> None:
    a, b = pair
    with np.load(b / 'settled.npz') as data:
        arrays = dict(data)
    arrays['grids'][3, 12, 5] = 2
    np.savez(b / 'settled.npz', **arrays)
    assert co.total_mismatches(co.compare_dirs(a, b)) == 1


def test_positive_control_jsonl_csv_changes_are_detected(pair: tuple[Path, Path]) -> None:
    a, b = pair
    lines = (b / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    lines[4] = json.dumps(dict(i=999))
    (b / 'events.jsonl').write_text('\n'.join(lines), encoding='utf-8')
    csv_lines = (b / 'review_data.csv').read_text(encoding='utf-8').splitlines()
    csv_lines[10] = '9,123.5'
    (b / 'review_data.csv').write_text('\n'.join(csv_lines), encoding='utf-8')
    report = co.compare_dirs(a, b)
    assert report['files']['events.jsonl']['mismatched'] == 1
    assert report['files']['review_data.csv']['mismatched_cells'] == 1  # 行 9 は先頭セルが同値で 2 列目だけ違う
    assert co.total_mismatches(report) == 2


def test_path_separator_only_difference_is_separated(pair: tuple[Path, Path]) -> None:
    a, b = pair
    with gzip.open(b / 'inputs.jsonl.gz', 'wt', encoding='utf-8') as stream:
        stream.write('\n'.join(json.dumps(dict(i=i, model_dir='models/v3')) for i in range(ROWS)))
    info = co.compare_dirs(a, b)['files']['inputs.jsonl.gz']
    assert info['mismatched'] == ROWS and info['path_separator_only'] == ROWS
    assert info['mismatched_excluding_path_separator'] == 0


def test_missing_file_or_length_difference_is_not_success(pair: tuple[Path, Path]) -> None:
    a, b = pair
    (b / 'events.jsonl').unlink()
    assert co.total_mismatches(co.compare_dirs(a, b)) == -1  # 比較できないものがある = 未確認
    (b / 'events.jsonl').write_text('\n'.join(json.dumps(dict(i=i)) for i in range(ROWS - 3)), encoding='utf-8')
    assert co.total_mismatches(co.compare_dirs(a, b)) == 3  # 行数差は不一致として数える


def test_empty_directories_are_not_success(tmp_path: Path) -> None:
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    assert co.total_mismatches(co.compare_dirs(tmp_path / 'a', tmp_path / 'b')) == -1
