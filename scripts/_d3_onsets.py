"""絵柄移動の確証時刻から、同じNEXTアニメーションの開始へ戻す。"""
from __future__ import annotations

import json
import os
from collections import Counter
import cv2
import numpy as np
from scripts._d3_inventory import OUT, SOURCES
from scripts._d3_observe import VIDEO_ROOT, ZENCHI
from scripts.enrich_e26_midchain import frame_at
from src.next_slide_detector import NextSlideDetector, DEFAULT_DIFF_THRESHOLD
from scripts._d3_candidates import load

LOOKBACK = 8
MAX_QUIET = 2
MERGE_SEC = .20
QUEUE_CONFIRM = 3


def queue_mode(rows: list[dict], side: int) -> tuple | None:
    """NEXTとDNEXTの4色を、欠測を除いた3回以上の一致で読む。"""
    counts: Counter = Counter()
    for row in rows:
        value = row['sides'][side]
        pair, double = value['next_pair'], value['dnext_pair']
        if pair and double and all(1 <= c <= 5 for c in pair+double):
            counts[tuple(pair+double)] += 1
    if not counts:
        return None
    value, count = counts.most_common(1)[0]
    return value if count >= QUEUE_CONFIRM else None


def supplement(source: str, scan: dict) -> None:
    """速い30fps動画で絵柄対応が飛ぶ場合を、NEXT列の物理的な繰上りで補う。"""
    rows = load(source)
    groups = json.loads((OUT/f'{source}_candidates.json').read_text())
    motions = scan['next_motions'][:]
    for side, group in enumerate(groups):
        for signal in group['signals']:
            if signal['type'] != 'NEXT' or signal['repeated']:
                continue
            index, stamp = signal['i'], signal['t']
            if any(e['side'] == side and abs(e['t']-stamp) <= MERGE_SEC for e in motions):
                continue
            before = queue_mode(rows[max(0, index-7):index], side)
            after = queue_mode(rows[index+3:index+9], side)
            if before is None or after is None or before == after or before[2:] != after[:2]:
                continue
            motions.append(dict(frame=round(stamp*scan['fps']), t=stamp, side=side,
                moving=True, queue_evidence=dict(before=before, after=after)))
    scan['next_motions'] = sorted(motions, key=lambda e: e['t'])


def activity(source: str, scan: dict) -> dict:
    """候補直前8原フレームのNEXT画素差を、既存の定義で計測する。"""
    video = ZENCHI if source == 'zenchi' else VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4'
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    rois = [NextSlideDetector(side=s)._get_rois() for s in ('1P', '2P')]
    requested: dict[int, set[int]] = {}
    for event in scan['next_motions']:
        for frame in range(max(0, event['frame']-LOOKBACK-1), event['frame']+1):
            requested.setdefault(frame, set()).add(event['side'])
    previous, result = {}, {}
    for number, (index, sides) in enumerate(sorted(requested.items())):
        frame = frame_at(cap, index)
        for side in sides:
            crops = [frame[y1:y2, x1:x2].astype(np.float32).mean(axis=2) for y1,y2,x1,x2 in rois[side]]
            if side in previous and previous[side][0] == index-1:
                old = previous[side][1]
                result[(index, side)] = float(np.mean([np.abs(a-b).mean() for a,b in zip(crops, old)]))
            previous[side] = index, crops
        if number % 2000 == 0:
            print(source, 'onset', number, len(requested), flush=True)
    cap.release()
    return result


def refine(scan: dict, diffs: dict) -> list[dict]:
    """動いた確証のあるアニメーションだけを、連続する画素変化の開始へ戻す。"""
    result = []
    for event in scan['next_motions']:
        onset, quiet = event['frame'], 0
        for index in range(event['frame'], event['frame']-LOOKBACK-1, -1):
            value = diffs.get((index, event['side']))
            if value is None:
                break
            quiet = quiet+1 if value < DEFAULT_DIFF_THRESHOLD else 0
            if quiet >= MAX_QUIET:
                break
            if value >= DEFAULT_DIFF_THRESHOLD:
                onset = index
        result.append(dict(event, detection_frame=event['frame'], detection_t=event['t'],
            frame=onset, t=onset/scan['fps'], onset_uncertainty_frames=event['frame']-onset))
    return result


def main() -> None:
    """原スキャンを保持して、開始時刻を別ファイルへ保存する。"""
    os.nice(19)
    cv2.setNumThreads(1)
    for source in (*SOURCES[:3], 'zenchi'):
        scan = json.loads((OUT/f'{source}_formula_scan.json').read_text())
        supplement(source, scan)
        diffs = activity(source, scan)
        scan['next_motions'] = refine(scan, diffs)
        (OUT/f'{source}_image_signals.json').write_text(json.dumps(scan, indent=2))


if __name__ == '__main__':
    main()
