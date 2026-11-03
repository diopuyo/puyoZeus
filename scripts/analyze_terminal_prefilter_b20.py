"""B20: scan_terminal_prefilter_b20 の npz から、低解像度予備判定の床の妥当性と決定同一性を集計する。"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

THRESHOLD = 0.55   # DEFAULT_NCC_THRESHOLD
CONFIRM = 2        # CONFIRM_FRAMES
FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
SCALES = {'half': (3, 4), 'quarter': (5, 6)}


def decisions(score: np.ndarray) -> np.ndarray:
    """従来検出器の確定 (連続CONFIRM回以上の閾値超え) をフレームごとに再現する。"""
    hit = score >= THRESHOLD
    streak = np.zeros(len(hit), dtype=int)
    for i in range(len(hit)):
        streak[i] = streak[i-1]+1 if hit[i] and i else int(hit[i])
    return streak >= CONFIRM


def emulate(full: np.ndarray, coarse: np.ndarray, floor: float) -> np.ndarray:
    """床未満なら粗いスコア、以上なら全解像度スコアを使う検出器の再現。"""
    return np.where(coarse < floor, coarse, full)


def main() -> None:
    files = sorted(glob.glob(sys.argv[1]))
    report = dict(files=[], per_scale={})
    data = {f: np.load(f)['data'] for f in files}
    for scale, (left, right) in SCALES.items():
        rows = []
        for floor in FLOORS:
            frames = diff = full_calls = pos = 0
            for f, d in data.items():
                for side, (c, full) in enumerate(((d[:, left], d[:, 1]), (d[:, right], d[:, 2]))):
                    est = emulate(full, c, floor)
                    a, b = decisions(full), decisions(est)
                    frames += len(full)
                    pos += int(a.sum())
                    diff += int((a != b).sum())
                    full_calls += int((c >= floor).sum())
            rows.append(dict(floor=floor, frame_sides=frames, decision_positive=pos,
                             decision_differences=diff, full_match_rate=full_calls/frames))
        report['per_scale'][scale] = rows
    allv = np.concatenate(list(data.values()))
    pos_mask = allv[:, 1:3] >= THRESHOLD
    report['frames_total'] = int(len(allv))
    report['threshold_hit_frame_sides'] = int(pos_mask.sum())
    for scale, (left, right) in SCALES.items():
        coarse = np.stack([allv[:, left], allv[:, right]], axis=1)
        report[scale] = dict(
            min_coarse_at_full_hit=float(coarse[pos_mask].min()) if pos_mask.any() else None,
            max_coarse_at_full_miss=float(coarse[~pos_mask].max()),
            p999_coarse_at_full_miss=float(np.percentile(coarse[~pos_mask], 99.9)),
            max_full_at_miss=float(allv[:, 1:3][~pos_mask].max()),
            # 全解像度が粗いスコアをどれだけ上回りうるか (床の根拠: 閾値-この値-余裕)
            max_full_minus_coarse=float((allv[:, 1:3]-coarse).max()),
            max_full_minus_coarse_when_full_ge_040=float(
                (allv[:, 1:3]-coarse)[allv[:, 1:3] >= 0.40].max()))
    for f, d in data.items():
        report['files'].append(dict(file=f, frames=int(len(d)), hit_p1=int((d[:, 1] >= THRESHOLD).sum()),
                                    hit_p2=int((d[:, 2] >= THRESHOLD).sum())))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
