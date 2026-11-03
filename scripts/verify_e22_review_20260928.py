"""E22納品動画とCSVで指定場面・死亡確定・試合境界を確認する。"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
from scripts.run_e3_exchange_eval_20260926 import save_json

DIRECTORY = Path('logs/review_zenchi_g41_43_e22')
OUT = Path('logs/e22')
SCENE = (2633.8, 2634.75)
TIMES = (2633.95, 2634.8, 2703.)


def main() -> None:
    """生成後の実CSVでも場面基準を満たすことを確認して静止画を保存する。"""
    status = json.loads((DIRECTORY/'status.json').read_text())
    command = status['command']
    start = float(command[command.index('--start-sec')+1])
    assert '--death-formula-guard' in command and '--death-candidate-guard' not in command
    with (DIRECTORY/'review_data.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    scene = [r for r in rows if SCENE[0] <= float(r['t_sec']) <= SCENE[1]]
    minimum = min(float(r['p1_selected']) for r in scene)
    assert minimum >= .85
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
    save_json(OUT/'REVIEW_QA.json', dict(scene_rows=len(scene), scene_min=minimum,
        video_rows=len(rows), complete=complete, images=images))


if __name__ == '__main__':
    main()
