"""完全な旧新の出力列を連続フレームで突合する。CP継ぎ目も厳密照合する。"""
from __future__ import annotations

import argparse
import gzip
from itertools import zip_longest
import json
from pathlib import Path
from typing import Iterator

from scripts.compare_image_free_all import compare, save_json
from scripts.image_free_segment_merge import directory_for, rows

ROUTES = ('old', 'new')


def validate(plan: dict) -> None:
    jobs = plan['jobs']
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('経路ジョブIDが重複しています')
    if sum(plan['worker_limits'].values()) > 8:
        raise ValueError('両PC合計8プロセスを超えています')
    for video, count in plan['expected_frames'].items():
        for route in ROUTES:
            selected = sorted((j for j in jobs if j['video_id'] == video and j['route'] == route),
                              key=lambda j: j['owned_start'])
            cursor = 0
            for job in selected:
                if (job['owned_start'] != cursor or not job['start_frame'] <= cursor < job['owned_end'] <= job['end_frame']
                        or job['start_frame'] > 0 and not job.get('seed_sha256')):
                    raise ValueError('経路の欠落・重複または前史CP欠落です')
                cursor = job['owned_end']
            if cursor != count:
                raise ValueError('経路が動画全体を被覆していません')
    if any(j['video_id'] not in plan['expected_frames'] or j['route'] not in ROUTES for j in jobs):
        raise ValueError('未知の動画または経路です')


def route_rows(jobs: list[dict], roots: list[Path]) -> Iterator[dict]:
    overlap = {}
    for job in sorted(jobs, key=lambda j: j['owned_start']):
        directory = directory_for(job, roots)
        progress = json.loads((directory/'progress.json').read_text())
        if not progress['complete'] or progress['frames'] != job['end_frame']:
            raise ValueError('単独経路が未完了です')
        cursor = job['start_frame']
        for row in rows(directory):
            if row['frame'] != cursor or cursor >= job['end_frame']:
                raise ValueError('単独経路のフレームが不連続です')
            if cursor in overlap and overlap.pop(cursor) != row:
                raise ValueError(f'通し先頭とCP再開の継ぎ目が不一致: {cursor}')
            if job['owned_start'] <= cursor < job['owned_end']:
                yield row
            elif cursor >= job['owned_end']:
                overlap[cursor] = row
            cursor += 1
        if cursor != job['end_frame']:
            raise ValueError('単独経路の出力が不足しています')
    if overlap:
        raise ValueError('CP継ぎ目の重複検算が完了していません')


def merge_video(video: str, count: int, jobs: list[dict], roots: list[Path], output: Path) -> dict:
    streams = [route_rows([j for j in jobs if j['video_id'] == video and j['route'] == route], roots)
               for route in ROUTES]
    state = dict(counts={}, display_scenes=0, previous_display_difference=False)
    stable, displayed, observed = [0, 0], [0, 0], 0
    with gzip.open(output/(video+'.differences.jsonl.gz'), 'wt', encoding='utf-8') as target:
        for index, pair in enumerate(zip_longest(*streams)):
            if any(r is None or r['frame'] != index for r in pair) or index >= count:
                raise ValueError('旧新経路の欠落・重複です')
            if any(pair[0][k] != pair[1][k] for k in ('fps', 'pts_ms', 'pixel_sha256')):
                raise ValueError(f'旧新の復号条件が不一致: {video}:{index}')
            values = [r['value'] for r in pair]
            for side, value in enumerate(values):
                stable[side] += int(value['states'] == ['STABLE', 'STABLE'])
                displayed[side] += int(value['displayed_probability'] is not None)
            difference = compare(values, state, index, pair[0]['fps'])
            if difference is not None:
                target.write(json.dumps(difference, ensure_ascii=False)+'\n')
            observed += 1
    if observed != count:
        raise ValueError('動画の最終フレームが不足しています')
    return dict(frames=observed, complete=True, differences=state['counts'],
        display_changed_scenes=state['display_scenes'], display_text_changed_scenes=state.get('display_text_scenes', 0),
        observed_feature_rows=state.get('observed_feature_rows', [0, 0]), observed_games=state.get('observed_games', [[], []]),
        both_stable_frames=stable, display_available_frames=displayed, continuous_history=True,
        missing_frames=0, duplicate_frames=0, decoded_pixels_equal=True)


def aggregate(plan: dict, roots: list[Path], output: Path) -> dict:
    validate(plan)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for video, count in plan['expected_frames'].items():
        reports[video] = merge_video(video, count, plan['jobs'], roots, output)
        save_json(output/(video+'.json'), reports[video])
    result = dict(comparison_mode='continuous_routes', videos=reports, complete=True,
        frames=sum(plan['expected_frames'].values()), gate_passed=False,
        verdict='fix2通し比較の全件分類・統括による品質判定待ち')
    save_json(output/'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    aggregate(json.loads(args.plan.read_text()), args.results, args.output)


if __name__ == '__main__':
    main()
