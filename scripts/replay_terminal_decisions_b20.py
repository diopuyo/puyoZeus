"""B20: 従来検出器(fast=False)と高速検出器(fast=True)の決定を、同一の実フレーム列で全数照合する。

実コード経路 (近傍照合・全域照合への落下を含む) の決定同一性を、母数つきで出す。
各フレームで両検出器へ同じ画像を渡し、confirmed (両側) と 側別の閾値超え判定を比較する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.exchange_event_terminal import ObservedDeathDetector
from src.match_end_detector import DEFAULT_NCC_THRESHOLD
from scripts.scan_terminal_prefilter_b20 import FRAMES, NATIVE_SIZE, STRIDE


def replay(video: str, start: float, end: float, output: Path) -> None:
    cv2.setNumThreads(1)
    slow, fast = ObservedDeathDetector(fast=False), ObservedDeathDetector(fast=True)
    cap = cv2.VideoCapture(FRAMES + video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(start * fps)
    last = min(int(end * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index, frames, diffs, hit_sides, examples = first, 0, 0, 0, []
    began = time.time()
    while index < last:
        ok, image = cap.read()
        if not ok:
            break
        for _ in range(STRIDE - 1):
            cap.grab()
        if (image.shape[1], image.shape[0]) != NATIVE_SIZE:
            image = cv2.resize(image, NATIVE_SIZE, interpolation=cv2.INTER_AREA)
        a, b = slow.update(image), fast.update(image)
        hits_a = tuple(s >= DEFAULT_NCC_THRESHOLD for s in slow.scores)
        hits_b = tuple(s >= DEFAULT_NCC_THRESHOLD for s in fast.scores)
        hit_sides += sum(hits_a)
        if a != b or hits_a != hits_b:
            diffs += 1
            examples.append(dict(frame=index, slow=list(a), fast=list(b), slow_scores=slow.scores[:],
                                 fast_scores=fast.scores[:]))
        frames += 1
        index += STRIDE
        if frames % 5000 == 0:
            print(video, index, frames, diffs, 'elapsed %.0fs' % (time.time() - began), flush=True)
    output.write_text(json.dumps(dict(video=video, start=start, end=end, frames=frames, differences=diffs,
        threshold_hit_frame_sides=hit_sides, fast_full_matches=fast.full_matches,
        fast_local_hits=fast.local_hits, examples=examples[:20]), indent=1))
    print('done', video, frames, 'diffs', diffs, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--end', type=float, default=1e9)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    replay(options.video, options.start, options.end, options.output)


if __name__ == '__main__':
    main()
