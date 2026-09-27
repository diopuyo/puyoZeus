"""未対応の確定盤面差を瞬間観測と照合し、判断用の原画像を保存する。"""
from __future__ import annotations

from collections import Counter
import numpy as np

from scripts.analyze_live_b5_alignment import load, events, pair_events, image_pair, OUTPUT, REPRESENTATIVES
from scripts.verify_live_b4_degradation import transcode, DEFAULT_VIDEO
from scripts.run_live_pipeline_20260928 import save_json


def inspect(left: dict, right: dict) -> dict:
    counts, candidates, seen = Counter(), [], set()
    for side in range(2):
        a, mapping = events(left, side)
        b, _ = events(right, side)
        matched = {i for i, _, _ in pair_events(a, b)}
        mask = (left['boards'][:, side] != right['boards'][:, side])
        mask &= (left['stable'][:, side] & right['stable'][:, side])[:, None, None]
        for frame, row, col in np.argwhere(mask):
            if mapping[frame] in matched:
                continue
            counts['all'] += 1
            if row == 0:
                counts['hidden'] += 1
                continue
            counts['visible'] += 1
            equal_raw = left['raw'][frame, side, row, col] == right['raw'][frame, side, row, col]
            equal_hsv = left['hsv_colors'][frame, side, row, col] == right['hsv_colors'][frame, side, row, col]
            counts['raw_agrees' if equal_raw else 'raw_differs'] += 1
            counts['hsv_agrees' if equal_hsv else 'hsv_differs'] += 1
            key = side, mapping[frame], int(row), int(col)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(dict(side=side, row=int(row), col=int(col), left_frame=int(frame),
                right_frame=int(frame), original=int(left['boards'][frame, side, row, col]),
                converted=int(right['boards'][frame, side, row, col]), raw_agrees=bool(equal_raw),
                hsv_agrees=bool(equal_hsv), event=mapping[frame],
                raw_original=int(left['raw'][frame, side, row, col]),
                raw_converted=int(right['raw'][frame, side, row, col])))
    return dict(counts=dict(counts), candidates=candidates)


def main() -> None:
    report, original = {}, load('original_a')
    for crf in (23, 35):
        result = inspect(original, load(f'crf{crf}'))
        # 瞬間観測も違う例を優先し、同じ確定盤面イベントの重複を避ける。
        candidates = sorted(result['candidates'], key=lambda c: c['raw_agrees'])
        selected, seen = [], set()
        for case in candidates:
            key = case['side'], case['event']
            if key not in seen:
                selected.append(case)
                seen.add(key)
            if len(selected) == REPRESENTATIVES:
                break
        video = OUTPUT / f'crf{crf}.mp4'
        transcode(DEFAULT_VIDEO, video, crf)
        try:
            for index, case in enumerate(selected):
                path = OUTPUT / f'crf{crf}_unresolved_{index+1}.png'
                image_pair(case, video, path)
                case['path'] = str(path)
        finally:
            video.unlink(missing_ok=True)
        result['representatives'] = selected
        report[str(crf)] = result
        save_json(OUTPUT / 'unresolved.json', report)


if __name__ == '__main__':
    main()
