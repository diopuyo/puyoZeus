"""固定画素の入力保証と、等価比較器の見逃し防止。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))
from compare_outputs import column_report, total_mismatches, union_rows
from equiv_frames import ARRAY_NAME, CachedCapture, META_NAME, digest


def cache(tmp_path: Path) -> CachedCapture:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    (tmp_path / ARRAY_NAME).write_bytes(frame.tobytes())
    meta = dict(first=10, last=11, total=100, fps=30., shape=list(frame.shape), sha256=[digest(frame)])
    (tmp_path / META_NAME).write_text(json.dumps(meta), encoding='utf-8')
    return CachedCapture(tmp_path, tmp_path / 'audit.json')


def test_cached_pixels_and_absolute_position(tmp_path: Path) -> None:
    capture = cache(tmp_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok, frame = capture.read()
    assert ok and frame.tolist() == np.arange(18).reshape(2, 3, 3).tolist()
    assert capture.get(cv2.CAP_PROP_POS_FRAMES) == 11
    capture.release()
    assert json.loads((tmp_path / 'audit.json').read_text())[0]['frame'] == 10


def test_cached_input_rejects_corruption_and_outside_range(tmp_path: Path) -> None:
    capture = cache(tmp_path)
    with pytest.raises(IndexError):
        capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 10)
    capture.stream.close()
    (tmp_path / ARRAY_NAME).write_bytes(bytes(18))
    capture.stream = (tmp_path / ARRAY_NAME).open('rb')
    with pytest.raises(ValueError, match='ハッシュ不一致'):
        capture.read()
    capture.release()


def test_incomplete_cache_copy_is_rejected(tmp_path: Path) -> None:
    capture = cache(tmp_path)
    capture.release()
    (tmp_path / ARRAY_NAME).write_bytes(b'partial')
    with pytest.raises(ValueError, match='複製未完了'):
        CachedCapture(tmp_path, tmp_path / 'audit.json')


def test_length_difference_is_not_lost_after_truncation() -> None:
    column = column_report(np.array([1]), np.array([1, 2]))
    assert column['length_diff'] == 1
    report = dict(missing=[], files={'display.npz': dict(rows_a=1, rows_b=2, columns={'x': column})})
    assert total_mismatches(report) == 1


def test_row_union_counts_more_than_one_thousand_differences() -> None:
    count = 1500
    column = column_report(np.zeros(count), np.ones(count))
    assert union_rows(dict(columns={'x': column})) == count


@pytest.mark.parametrize('name,info', [
    ('display.npz', dict(rows_a=1, rows_b=1, missing_columns=['p1'])),
    ('review_data.csv', dict(header_equal=False)),
    ('review_data.csv', dict(length_diff=1)),
])
def test_structural_differences_cannot_pass(name: str, info: dict) -> None:
    assert total_mismatches(dict(missing=[], files={name: info})) > 0
