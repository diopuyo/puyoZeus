"""保存済みの境界原本からB15の事前母数を固定する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.measure_realtime_breakdown_20260928 import ASSETS, DEFAULT_VIDEO
from scripts.measure_live_b6 import save

START, DURATION = 2580.566, 3600.0
END = START+DURATION
ROOT = ASSETS/'data/verify/zenchi_two_sets_review_source_2026-08-31'
DELIVERY = ASSETS/'data/verify/zenchi_two_sets_redesign_review_delivery_v2_2026-08-31/DELIVERY.json'
OUTPUT = Path('logs/live_b15')
MS = 1000.0
SET_TRANSITION = 3626.0


def boundary_rows(path: Path, number: int) -> list[dict]:
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assignments = {r['payload']['local_game_index']: r['payload']['official_game_number']
                   for r in events if r.get('event_type') == 'official_game_assignment'}
    rows = [dict(set=number, local_game=0, t_sec=0.0 if number == 1 else SET_TRANSITION)]
    for event in events:
        if event.get('event_type') == 'match_boundary_evidence':
            rows.append(dict(set=number, local_game=event['payload']['opening_game_index_unverified'],
                             t_sec=event['timing']['occurred_earliest_ms']/MS,
                             event_id=event['event_id']))
    for row in rows:
        row['official_game'] = assignments.get(row['local_game'])
    return rows


def select_games(rows: list[dict], start: float, end: float) -> list[dict]:
    selected = [row for row in rows if start <= row['t_sec'] < end]
    previous = [row for row in rows if row['t_sec'] < start]
    if previous and (not selected or selected[0]['t_sec'] > start):
        first = max(previous, key=lambda row: row['t_sec'])
        selected.insert(0, dict(first, t_sec=start, original_start=first['t_sec'], left_censored=True))
    return selected


def prepare(output: Path = OUTPUT) -> dict:
    import cv2
    delivery = json.loads(DELIVERY.read_text())
    tier = [s for s in delivery['limitations'] if 'チャレンジャー級' in s]
    if not tier:
        raise ValueError('保存済みの上級者ティア証跡が必要です')
    sources, rows = [], []
    for number in (1, 2):
        run = Path(delivery['event_runs'][str(number)])
        path = run/'events/part-00000.jsonl'
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        snapshot = next((ROOT/'work').glob(f'zenchi_set{number}_*.npz'))
        with np.load(snapshot) as data:
            observed = set(data['game_idx'].tolist())
        # set1末尾の「次を開く」だけの境界をset2先頭と二重計上しない。
        rows.extend(r for r in boundary_rows(path, number) if r['local_game'] in observed)
    selected = select_games(rows, START, END)
    capture = cv2.VideoCapture(str(DEFAULT_VIDEO))
    fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if fps <= 0 or count/fps < END:
        raise ValueError('動画の60分連続区間を確保できません')
    result = dict(start=START, end=END, duration=DURATION, video=str(DEFAULT_VIDEO), fps=fps,
        video_frames=int(count), tier_evidence=tier, tier_source=str(DELIVERY), sources=sources,
        expected_games=len(selected), expected_internal_boundaries=len(selected)-1, games=selected,
        basis='既存match_boundary_evidence。公式番号未割当も隠さず候補として保持する。',
        inter_set_note='約3490〜3626秒に結果・再選択画面。動画もプロセスも切らずに通過する。',
        comparison_windows=[[START, START+400], [START+1600, START+2000], [END-400, END]])
    save(output/'plan.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
