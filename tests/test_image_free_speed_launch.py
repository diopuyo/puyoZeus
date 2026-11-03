"""段3未合格・改変済み成果物の誤起動を止める。"""
from pathlib import Path
import json

import pytest

from scripts.image_free_speed_launch import sha, verify, verify_segmentation


def test_unpassed_gate_fails_before_reading_assets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='未合格'):
        verify({}, tmp_path, {'gate_passed': False})


def test_changed_source_rejected(tmp_path: Path) -> None:
    (tmp_path/'src').mkdir()
    (tmp_path/'src/module.py').write_text('changed')
    with pytest.raises(ValueError, match='ソース'):
        verify({}, tmp_path, {'gate_passed': True, 'source_manifest': {}})


def segmented_gate(tmp_path: Path) -> tuple[dict, dict]:
    segments = [dict(id=f's{i}', video_id='v', start_frame=i*10, owned_start=i*10,
                     end_frame=(i+1)*10, fps=60., host='pc1') for i in range(3)]
    proof = dict(passed=True, cases=[dict(video='v', start=i*10, passed=True,
        complete_reference=True, mismatches=[], feature_rows=2) for i in range(3)])
    plan = dict(segments=segments, expected_frames={'v': 30})
    (tmp_path/'split_proof.json').write_text(json.dumps(proof))
    (tmp_path/'segment_plan.json').write_text(json.dumps(plan))
    gate = dict(split_proof_sha256=sha(tmp_path/'split_proof.json'),
                segment_plan_sha256=sha(tmp_path/'segment_plan.json'))
    return dict(output=str(tmp_path), host='pc1', videos=segments), gate


def test_verified_segments_and_assignment(tmp_path: Path) -> None:
    config, gate = segmented_gate(tmp_path)
    verify_segmentation(config, gate)
    config['videos'] = config['videos'][:-1]
    with pytest.raises(ValueError, match='割当'):
        verify_segmentation(config, gate)


@pytest.mark.parametrize('change', ['same_boundary', 'sparse', 'mismatch', 'empty'])
def test_invalid_proof_is_rejected(tmp_path: Path, change: str) -> None:
    config, gate = segmented_gate(tmp_path)
    path = tmp_path/'split_proof.json'
    proof = json.loads(path.read_text())
    case = proof['cases'][-1]
    if change == 'same_boundary':
        case['start'] = 0
    elif change == 'sparse':
        case['complete_reference'] = False
    elif change == 'mismatch':
        case['mismatches'] = [{'column': 'board'}]
    else:
        case['feature_rows'] = 0
    path.write_text(json.dumps(proof))
    gate['split_proof_sha256'] = sha(path)
    with pytest.raises(ValueError, match='未合格'):
        verify_segmentation(config, gate)


def test_modified_plan_is_rejected(tmp_path: Path) -> None:
    config, gate = segmented_gate(tmp_path)
    (tmp_path/'segment_plan.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA'):
        verify_segmentation(config, gate)
