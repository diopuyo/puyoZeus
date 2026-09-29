"""E23納品CSVの初到達時刻と動画の指定場面を検証する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.run_e23_multilanding_20260928 import SCENE, P2_MAX

DIRECTORY = Path('logs/review_zenchi_g41_43_e23')
OUT = Path('logs/e23')
TIMES = (2770.4, 2770.8, 2774.)


def main() -> None:
    """実際に描画したCSVと圧縮済み動画を確認し、目視用画像を保存する。"""
    status = json.loads((DIRECTORY/'status.json').read_text())
    command = status['command']
    start = float(command[command.index('--start-sec')+1])
    assert '--death-formula-guard' in command and '--multi-landing-death' in command
    assert '--death-candidate-guard' not in command
    with (DIRECTORY/'review_data.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    scene = [r for r in rows if SCENE[0] <= float(r['t_sec']) <= SCENE[1]]
    first = next(float(r['t_sec']) for r in scene if 1-float(r['p1_display']) <= P2_MAX)
    metrics = json.loads((OUT/'on/METRICS.json').read_text())
    assert first < metrics['scene']['baseline_first_sec']
    complete = json.loads((DIRECTORY/'complete.json').read_text())
    cap = cv2.VideoCapture(complete['mobile'][0]['path'])
    images = []
    for stamp in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, (stamp-start)*1000)
        ok, frame = cap.read()
        assert ok
        path = OUT/f'review_{stamp:.2f}.jpg'
        assert cv2.imwrite(str(path), frame)
        images.append(str(path))
    cap.release()
    save_json(OUT/'REVIEW_QA.json', dict(scene_rows=len(scene), first_sec=first,
        replay_first_sec=metrics['scene']['first_sec'], video_rows=len(rows), complete=complete, images=images))


if __name__ == '__main__':
    main()
