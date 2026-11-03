"""旧CPの継ぎ目と旧新フレームの対応を、欠落・改変例で検証する。"""
import gzip
import json
from pathlib import Path

import pytest

from scripts.image_free_stream_merge import aggregate


def fixture(root: Path, fault: str = '') -> dict:
    jobs = [dict(id='head', route='old', start_frame=0, end_frame=3, owned_start=0, owned_end=2),
            dict(id='tail', route='old', start_frame=2, end_frame=4, owned_start=2, owned_end=4, seed_sha256='cp'),
            dict(id='new', route='new', start_frame=0, end_frame=4, owned_start=0, owned_end=4)]
    for job in jobs:
        job['video_id'] = 'v'
        directory = root/job['id']
        directory.mkdir()
        records = []
        for index in range(job['start_frame'], job['end_frame']):
            value = dict(states=['STABLE', 'STABLE'], displayed_probability=.5,
                display_text=51 if job['route'] == 'new' and index in (1, 2) else 50,
                b3_features={'lean_row_p1': index}, boards=[[0]])
            if fault == 'seam' and job['id'] == 'tail' and index == 2:
                value['boards'] = [[1]]
            records.append(dict(frame=index, fps=60, pts_ms=index*1000/60,
                pixel_sha256='wrong' if fault == 'pixels' and job['id'] == 'new' else 'same', value=value))
        with gzip.open(directory/'rows.gz', 'wt') as stream:
            json.dump(dict(rows=records), stream)
        (directory/'checkpoint.json').write_text(json.dumps(dict(chunks=['rows.gz'])))
        (directory/'progress.json').write_text(json.dumps(dict(frames=job['end_frame'], complete=True)))
    return dict(jobs=jobs, expected_frames={'v': 4}, worker_limits={'pc1': 2, 'pc2': 6})


def test_continuous_scene_across_cp(tmp_path: Path) -> None:
    result = aggregate(fixture(tmp_path), [tmp_path], tmp_path/'out')
    video = result['videos']['v']
    assert video['display_text_changed_scenes'] == 1
    assert video['differences'] == {'display_text': 2}
    assert video['frames'] == 4 and video['missing_frames'] == video['duplicate_frames'] == 0
    assert video['both_stable_frames'] == [4, 4]


@pytest.mark.parametrize('fault,match', [('seam', '継ぎ目'), ('pixels', '復号条件')])
def test_invalid_stream_fails(tmp_path: Path, fault: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        aggregate(fixture(tmp_path, fault), [tmp_path], tmp_path/'out')


def test_missing_owned_frame_fails(tmp_path: Path) -> None:
    plan = fixture(tmp_path)
    plan['jobs'][1]['owned_start'] = 3
    with pytest.raises(ValueError, match='欠落'):
        aggregate(plan, [tmp_path], tmp_path/'out')


def test_prefix_can_move_to_other_pc_without_changing_coverage(tmp_path: Path) -> None:
    pc1, pc2 = tmp_path/'pc1', tmp_path/'pc2'
    pc1.mkdir()
    pc2.mkdir()
    plan = fixture(pc1)
    plan['jobs'][0]['host'] = 'pc2'
    (pc1/'head').rename(pc2/'head')
    result = aggregate(plan, [pc1, pc2], tmp_path/'out')['videos']['v']
    assert result['frames'] == 4
    assert result['missing_frames'] == result['duplicate_frames'] == 0
    assert result['display_text_changed_scenes'] == 1


def test_abandoned_pc1_copy_must_not_be_in_merge_inputs(tmp_path: Path) -> None:
    import shutil
    pc1, pc2 = tmp_path/'pc1', tmp_path/'pc2'
    pc1.mkdir()
    pc2.mkdir()
    plan = fixture(pc1)
    shutil.copytree(pc1/'head', pc2/'head')
    with pytest.raises(ValueError, match='重複'):
        aggregate(plan, [pc1, pc2], tmp_path/'out')
