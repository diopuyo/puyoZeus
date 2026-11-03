"""対の比較では履歴を繋がず、既知窓の母数を要求する。"""
import io
import json
from pathlib import Path

import pytest

from scripts import image_free_paired as paired
from scripts.image_free_speed_launch import verify_paired


def test_paired_preserves_local_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path/'progress.json').write_text(json.dumps(dict(complete=True, frames=11)))
    row = dict(frame=10, old=dict(b3_features={'lean_row_p1': 0}, display_text=50),
               new=dict(b3_features={'lean_row_p1': 1}, display_text=51))
    monkeypatch.setattr(paired, 'rows', lambda path: iter([row]))
    stream = io.StringIO()
    report = paired.paired_segment(dict(id='s', owned_start=10, end_frame=11, fps=60), tmp_path, stream)
    assert report['differences'] == {'b3_features': 1, 'display_text': 1}
    assert json.loads(stream.getvalue())['old']['b3_features']['lean_row_p1'] == 0


@pytest.mark.parametrize('missing,changed', [(False, False), (True, False), (False, True)])
def test_known_windows_require_all_frames(monkeypatch: pytest.MonkeyPatch, missing: bool, changed: bool) -> None:
    monkeypatch.setattr(paired, 'KNOWN_WINDOWS', {'c68': (10, 12)})
    records = [dict(frame=i, old={'boards': [[0]]}, new={'boards': [[int(changed)]]}) for i in range(10, 13)]
    monkeypatch.setattr(paired, 'rows', lambda path: iter(records[:-1] if missing else records))
    result = paired.verify_known(dict(output='unused', videos=[dict(id='s', video_id='c68',
        owned_start=10, end_frame=13)]))
    assert result['passed'] == (not missing and not changed)
    assert result['continuous_equivalence'] is False


def test_gate_requires_r2_and_known_windows() -> None:
    proof = dict(comparison_mode=paired.COMPARISON_MODE, disclaimer=paired.DISCLAIMER,
        continuous_equivalence=False, passed=True, cases=[dict(video=v, start=a, end=b+1,
        frames=b-a+1, passed=True, board_mismatches=[]) for v, (a, b) in paired.KNOWN_WINDOWS.items()])
    with pytest.raises(ValueError, match='後半'):
        verify_paired(proof, {})
    gate = dict(fix2_r2_exit=0, fix2_r2_commit='sha')
    verify_paired(proof, gate)
    proof['cases'][0]['frames'] -= 1
    with pytest.raises(ValueError, match='盤面差0'):
        verify_paired(proof, gate)
