"""通しとの同等性を主張しない、同一起点・同一助走の旧新対比較。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.compare_image_free_all import compare, save_json
from scripts.image_free_segment_merge import directory_for, rows, validate_segments

COMPARISON_MODE = 'paired_cold'
DISCLAIMER = '通しとの同等性は不成立・対の比較'
KNOWN_WINDOWS = {'c68': (10296, 10622), 'video_d01': (10658, 10763)}


def paired_segment(segment: dict, directory: Path, stream: object) -> dict:
    progress = json.loads((directory/'progress.json').read_text())
    if not progress['complete'] or progress['frames'] != segment['end_frame']:
        raise ValueError(f"未完了区間: {segment['id']}")
    state = dict(counts={}, display_scenes=0, previous_display_difference=False)
    cursor = segment['owned_start']
    for row in rows(directory):
        if row['frame'] != cursor or cursor >= segment['end_frame']:
            raise ValueError('対の比較の所有フレームが不連続です')
        difference = compare([row['old'], row['new']], state, cursor, segment['fps'])
        if difference is not None:
            stream.write(json.dumps(dict(segment_id=segment['id'], **difference), ensure_ascii=False)+'\n')
        cursor += 1
    if cursor != segment['end_frame']:
        raise ValueError('対の比較のフレームが不足しています')
    return dict(id=segment['id'], frames=cursor-segment['owned_start'], differences=state['counts'],
        display_changed_scenes=state['display_scenes'],
        display_text_changed_scenes=state.get('display_text_scenes', 0),
        observed_feature_rows=state.get('observed_feature_rows', [0, 0]),
        observed_games=state.get('observed_games', [[], []]), complete=True)


def aggregate(manifest: dict, roots: list[Path], output: Path) -> dict:
    validate_segments(manifest['segments'], manifest['expected_frames'])
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    with gzip.open(output/'paired_differences.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for segment in sorted(manifest['segments'], key=lambda s: (s['video_id'], s['owned_start'])):
            reports.append(paired_segment(segment, directory_for(segment, roots), stream))
    counts = {}
    for report in reports:
        for key, value in report['differences'].items():
            counts[key] = counts.get(key, 0)+value
    result = dict(comparison_mode=COMPARISON_MODE, disclaimer=DISCLAIMER,
        continuous_equivalence=False, complete=True, gate_passed=False, segments=reports,
        frames=sum(r['frames'] for r in reports), differences=counts,
        display_changed_scenes=sum(r['display_changed_scenes'] for r in reports),
        display_text_changed_scenes=sum(r['display_text_changed_scenes'] for r in reports),
        scene_definition='区間内の連続差分を1場面とし、区間間では接続しない')
    save_json(output/'result.json', result)
    return result


def verify_known(config: dict) -> dict:
    """独立検収の既知2窓を全フレーム含み、盤面差0の再現を要求する。"""
    cases = []
    for video, (first, last) in KNOWN_WINDOWS.items():
        candidates = [s for s in config['videos'] if s['video_id'] == video
                      and s['owned_start'] <= first and s['end_frame'] > last]
        if len(candidates) != 1:
            raise ValueError('既知区間を一意に被覆していません')
        segment = candidates[0]
        checked, mismatches = [], []
        for row in rows(Path(config['output'])/segment['id']):
            if first <= row['frame'] <= last:
                checked.append(row['frame'])
                if row['old']['boards'] != row['new']['boards']:
                    mismatches.append(dict(frame=row['frame'], old=row['old']['boards'], new=row['new']['boards']))
        complete = checked == list(range(first, last+1))
        cases.append(dict(video=video, start=first, end=last+1, frames=len(checked),
            segment=segment, board_mismatches=mismatches, passed=complete and not mismatches))
    return dict(comparison_mode=COMPARISON_MODE, disclaimer=DISCLAIMER,
                continuous_equivalence=False, cases=cases, passed=all(c['passed'] for c in cases))
