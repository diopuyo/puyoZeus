"""連続OCR比較が途中終了・フレーム欠落・不一致を合格にしないことを検査する。"""
import json
from pathlib import Path

import pytest

from scripts.compare_image_free_digit_stream import write_report


@pytest.mark.parametrize('frames,expected,complete,equal,passed', [
    (10, 10, True, 20, True),
    (10, 10, False, 20, False),
    (9, 10, True, 18, False),
    (10, 10, True, 19, False),
    (0, 0, True, 0, False),
])
def test_completion_gate(tmp_path: Path, frames: int, expected: int,
                         complete: bool, equal: int, passed: bool) -> None:
    counts = {name: [equal, frames * 2] for name in ('digits', 'scores', 'formulas')}
    write_report(tmp_path, counts, frames, expected, 0, complete)
    report = json.loads((tmp_path / 'comparison.json').read_text(encoding='utf-8'))
    assert report['gate_passed'] is passed
    assert '未測定' in report['downstream']
