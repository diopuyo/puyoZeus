"""全項目の検証器が差分を列別に残し、既知の不一致を見逃さない。"""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import image_free_split_verify as probe


@pytest.mark.parametrize('wrong_feature', [False, True])
def test_full_reference_checks_real_feature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           wrong_feature: bool) -> None:
    from scripts import image_free_runtime
    baseline = tmp_path/'reference.gz'
    feature = dict(video_id='live', game_idx=4, frame_idx=10, orientation=0,
                   row_key='live|g4|f10|o0', lean_row_p1=9, lean_row_p2=-1, score=100)
    value = dict(b3_features=feature)
    counts = dict(lean_rows=10, games=4, adopted_games=4)
    with gzip.open(baseline, 'wt') as stream:
        json.dump(dict(rows=[dict(frame=10, value=value, counters=counts)]), stream)
    actual = dict(b3_features=dict(feature, game_idx=1, lean_row_p1=2,
                                  row_key='live|g1|f10|o0', score=101 if wrong_feature else 100))
    monkeypatch.setattr(image_free_runtime, 'Runtime', lambda *args: object())
    monkeypatch.setattr(probe, 'frames', lambda *args: [(10, None, 60.)])
    monkeypatch.setattr(probe, 'observed', lambda *args: actual)
    monkeypatch.setattr(probe, 'counters', lambda *args: dict(lean_rows=3, games=1, adopted_games=1))
    case = dict(id='boundary', video='v', warm_start=10, start=10, end=11)
    result = probe.verify(case, dict(videos=[dict(id='v', path='unused')]), baseline,
                          tmp_path, SimpleNamespace(checkpoint=lambda: None))
    assert result['passed'] is not wrong_feature
    assert result['feature_rows'] == 1 and result['complete_reference']
    if wrong_feature:
        assert result['mismatches'][0]['columns'] == [dict(column='b3_features.score', old=100, split=101)]
