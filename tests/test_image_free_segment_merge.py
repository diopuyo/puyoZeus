"""分割の接合がメタ列と場面境界を通し実行と一致させる。"""
import gzip
import io
import json
from pathlib import Path

import pytest

from scripts.compare_image_free_all import compare
from scripts.image_free_segment_merge import aggregate, validate_segments


def values(index: int, lean_offset: int = 0, game_offset: int = 0) -> dict:
    return dict(b3_features=dict(video_id='live', game_idx=index//3-game_offset,
        frame_idx=index, orientation=0, row_key=f'live|g{index//3-game_offset}|f{index}|o0',
        lean_row_p1=index-lean_offset, lean_row_p2=-1, score_p1=100),
        displayed_probability=0.5, display_text=50)


def write_segment(root: Path, segment: dict, offset: int) -> None:
    directory = root/segment['id']
    directory.mkdir()
    first, end = segment['owned_start'], segment['end_frame']
    rows = []
    for index in range(first, end):
        old = values(index, offset)
        new = dict(old, display_text=51 if 1 <= index <= 4 else 50,
                   displayed_probability=.51 if 1 <= index <= 4 else .5)
        rows.append(dict(frame=index, old=old, new=new))
    with gzip.open(directory/'rows.gz', 'wt') as stream:
        json.dump(dict(rows=rows), stream)
    (directory/'checkpoint.json').write_text(json.dumps(dict(chunks=['rows.gz'])))
    (directory/'progress.json').write_text(json.dumps(dict(complete=True, frames=end,
        owned_counters=[dict(lean_rows=first-offset, games=0)]*2,
        end_counters=[dict(lean_rows=end-offset, games=0)]*2)))


def test_join_keeps_one_scene_across_boundary(tmp_path: Path) -> None:
    segments = [dict(id='a', video_id='v', start_frame=0, owned_start=0, end_frame=3, fps=60),
                dict(id='b', video_id='v', start_frame=1, owned_start=3, end_frame=6, fps=60)]
    write_segment(tmp_path, segments[0], 0)
    write_segment(tmp_path, segments[1], 2)
    result = aggregate(dict(segments=segments, expected_frames={'v': 6}), [tmp_path], tmp_path/'out')
    reference = dict(counts={}, display_scenes=0, previous_display_difference=False)
    for index in range(6):
        old = values(index)
        new = dict(old, display_text=51 if 1 <= index <= 4 else 50,
                   displayed_probability=.51 if 1 <= index <= 4 else .5)
        compare([old, new], reference, index, 60)
    actual = result['videos']['v']
    assert actual['differences'] == reference['counts']
    assert actual['display_changed_scenes'] == actual['display_text_changed_scenes'] == 1
    assert result['frames'] == 6


@pytest.mark.parametrize('start,end', [(1, 5), (0, 4), (0, 6), (0, 0)])
def test_gaps_and_wrong_coverage_fail(start: int, end: int) -> None:
    with pytest.raises(ValueError):
        validate_segments([dict(id='a', video_id='v', start_frame=0, owned_start=start,
                                end_frame=end)], {'v': 5})


def test_duplicate_segment_id_fails() -> None:
    segment = dict(id='a', video_id='v', start_frame=0, owned_start=0, end_frame=5)
    with pytest.raises(ValueError, match='重複'):
        validate_segments([segment, segment], {'v': 5})
