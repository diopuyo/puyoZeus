"""元動画の全フレームから掛け算記号の出現・消滅を観測する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from scripts._d3_inventory import OUT, SOURCES
from scripts._d3_observe import VIDEO_ROOT, ZENCHI

NICE = 19
PAD_Y = 16
PAD_X = 8
MULT_LEFT = 160
MULT_RIGHT = 200
PROGRESS = 6000
NEXT_DEBOUNCE_SEC = .20


def multiply(frame: Any, template: Any) -> list[float]:
    """既存×テンプレート・既存閾値を使い、上下の欠けだけを避ける。"""
    import cv2
    from src.score_ocr import SCORE_1P_REGION, SCORE_2P_REGION
    values = []
    for y1, y2, x1, _ in (SCORE_1P_REGION, SCORE_2P_REGION):
        roi = frame[y1-PAD_Y:y2+PAD_Y, x1+MULT_LEFT-PAD_X:x1+MULT_RIGHT+PAD_X]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        values.append(float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max()))
    return values


def run(source: str, template: Any, threshold: float) -> None:
    """フレーム間引きをせず全原動画フレームの式表示を保存する。"""
    import cv2
    from scripts.enrich_e26_midchain import frame_at
    from scripts._d3_next_probe import next_translation
    video = ZENCHI if source == 'zenchi' else VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4'
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    fps = cap.get(cv2.CAP_PROP_FPS)
    first, last = ((2580.0, 3427.2) if source == 'zenchi' else (0, 900))
    results, active, start = [], [None, None], time.monotonic()
    motions, previous, last_motion = [], None, [-1.0, -1.0]
    for index in range(round(first*fps), min(round(last*fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        frame = frame_at(cap, index)
        values = multiply(frame, template)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            for side in range(2):
                motion = next_translation(previous, gray, side)
                if motion['moving']:
                    if index/fps-last_motion[side] > NEXT_DEBOUNCE_SEC:
                        motions.append(dict(frame=index, t=index/fps, side=side, **motion))
                    last_motion[side] = index/fps
        previous = gray
        for side, value in enumerate(values):
            if value >= threshold and active[side] is None:
                active[side] = dict(frame=index, t=index/fps, side=side, ncc=value)
            if value < threshold and active[side] is not None:
                results.append(dict(**active[side], end_frame=index, end=index/fps))
                active[side] = None
        if index % PROGRESS == 0:
            print(source, index/fps, len(results), round(time.monotonic()-start, 1), flush=True)
    for event in active:
        if event is not None:
            results.append(dict(**event, end_frame=index+1, end=(index+1)/fps))
    cap.release()
    (OUT/f'{source}_formula_scan.json').write_text(json.dumps(dict(fps=fps, intervals=results, next_motions=motions), indent=2))


def main() -> None:
    """先行画像観測の完了後だけ、nice 19・並列1でスキャンする。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    options = parser.parse_args()
    if not options.worker:
        with (OUT/'formula_scan.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-B', '-m', 'scripts._d3_formula_scan', '--worker'],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1'))
        print(child.pid)
        return
    os.nice(NICE)
    while not (OUT/'zenchi_observations_complete.json').exists():
        time.sleep(5)
    import cv2
    from src.score_ocr import ScoreOcr, FORMULA_MULT_NCC_MIN
    cv2.setNumThreads(1)
    template = ScoreOcr.load_default()._mult_template_gray
    for source in (*SOURCES[:3], 'zenchi'):
        run(source, template, FORMULA_MULT_NCC_MIN)


if __name__ == '__main__':
    main()
