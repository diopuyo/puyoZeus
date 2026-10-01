"""5C採点の原票欠落と、変更前実装の復元を検証する。"""
import json
from pathlib import Path

import pytest

from scripts import prefire_v5c_report as report
from scripts import prefire_v5c_profile as profile


def test_verification_rejects_missing_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        report.verification()


def test_verification_counts_all_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, 'OUT', tmp_path)
    root = tmp_path/'final_verification'
    root.mkdir()
    for index in range(report.SAMPLES):
        row = dict(transitions=4, mismatches=int(index == 0), feature_boards=2, feature_mismatches=0,
                   max_value_error=0., dependencies=[], values=[dict(missed=False, table_sequential_equal=True)])
        (root/f'case_{index:03d}.json').write_text(json.dumps(row))
    result = report.verification()
    assert result['samples'] == 100
    assert result['transitions'] == 400
    assert result['mismatches'] == 1


def test_baseline_profile_uses_pinned_source() -> None:
    layer = profile.baseline_layer()
    assert layer.__module__ == 'baseline_prefire_best_play_v5'
    assert not hasattr(layer, '_compute_v5')
    assert layer._schedule_v5.__globals__['exhaustive'].__name__ == 'baseline_prefire_v5b_search'
