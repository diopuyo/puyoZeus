"""所有区間を順番に接合し、通し実行と同じ比較関数で場面数を集計する。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator

from scripts.compare_image_free_all import compare, save_json
from scripts.image_free_metadata import MetadataOffset


def validate_segments(segments: list[dict], expected: dict[str, int]) -> None:
    if len({s['id'] for s in segments}) != len(segments):
        raise ValueError('区間IDが重複しています')
    for video, end in expected.items():
        ordered = sorted((s for s in segments if s['video_id'] == video), key=lambda s: s['owned_start'])
        cursor = 0
        for segment in ordered:
            if (segment['owned_start'] != cursor or segment['end_frame'] <= cursor
                    or not 0 <= segment['start_frame'] <= cursor):
                raise ValueError(f'所有区間の欠落・重複: {video}:{cursor}')
            cursor = segment['end_frame']
        if cursor != end:
            raise ValueError(f'全フレームを被覆していません: {video}:{cursor}/{end}')
    if any(s['video_id'] not in expected for s in segments):
        raise ValueError('未知の動画区間です')


def rows(directory: Path) -> Iterator[dict]:
    manifest = json.loads((directory/'checkpoint.json').read_text())
    for name in manifest['chunks']:
        with gzip.open(directory/name, 'rt', encoding='utf-8') as stream:
            yield from json.load(stream)['rows']


def directory_for(segment: dict, roots: list[Path]) -> Path:
    candidates = [root/segment['id'] for root in roots if (root/segment['id']/'progress.json').exists()]
    if len(candidates) != 1:
        raise ValueError(f"区間結果が欠落または重複: {segment['id']}")
    return candidates[0]


def consume(segment: dict, directory: Path, state: dict, totals: list[dict], stream: object) -> dict:
    progress = json.loads((directory/'progress.json').read_text())
    if not progress['complete'] or progress['frames'] != segment['end_frame']:
        raise ValueError(f"未完了区間: {segment['id']}")
    offsets = [MetadataOffset(**{k: totals[i][k]-v for k, v in start.items()},
                              previous_adopted=state.get('last_adopted', [None, None])[i])
               for i, start in enumerate(progress['owned_counters'])]
    cursor = segment['owned_start']
    for row in rows(directory):
        if row['frame'] != cursor or cursor >= segment['end_frame']:
            raise ValueError('区間内のフレームが不連続です')
        values = [offset.apply(row[name]) for name, offset in zip(('old', 'new'), offsets)]
        difference = compare(values, state, cursor, segment['fps'])
        if difference is not None:
            stream.write(json.dumps(difference, ensure_ascii=False)+'\n')
        cursor += 1
    if cursor != segment['end_frame']:
        raise ValueError('区間のフレーム数が不足しています')
    state['last_adopted'] = [v.get('game', {}).get('adopted') for v in values]
    for i, end in enumerate(progress['end_counters']):
        for key, value in end.items():
            if value < progress['owned_counters'][i][key]:
                raise ValueError('区間内で通し番号が巻き戻りました')
            totals[i][key] += value-progress['owned_counters'][i][key]
    return dict(id=segment['id'], offsets=[vars(o) for o in offsets], frames=cursor-segment['owned_start'])


def aggregate(manifest: dict, roots: list[Path], output: Path) -> dict:
    if manifest.get('comparison_mode') == 'paired_cold':
        from scripts.image_free_paired import aggregate as paired_aggregate
        return paired_aggregate(manifest, roots, output)
    segments, expected = manifest['segments'], manifest['expected_frames']
    validate_segments(segments, expected)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for video, count in expected.items():
        state = dict(counts={}, display_scenes=0, previous_display_difference=False)
        totals = [dict(lean_rows=0, games=0, adopted_games=0) for _ in range(2)]
        audits = []
        with gzip.open(output/(video+'.differences.jsonl.gz'), 'wt', encoding='utf-8') as stream:
            for segment in sorted((s for s in segments if s['video_id'] == video), key=lambda s: s['owned_start']):
                audits.append(consume(segment, directory_for(segment, roots), state, totals, stream))
        reports[video] = dict(frames=count, expected_frames=count, complete=True,
            differences=state['counts'], display_changed_scenes=state['display_scenes'],
            display_text_changed_scenes=state.get('display_text_scenes', 0),
            observed_feature_rows=state.get('observed_feature_rows'), segments=audits)
        reports[video]['observed_games'] = state.get('observed_games', [[], []])
        save_json(output/(video+'.json'), reports[video])
    result = dict(videos=reports, complete=True, gate_passed=False,
                  frames=sum(expected.values()), canonical_aggregation=True)
    save_json(output/'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    aggregate(json.loads(args.manifest.read_text()), args.results, args.output)


if __name__ == '__main__':
    main()
