"""3動画で保留された発火を原映像の時系列で点検する。"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT

OUT = Path('logs/e22/hold_frames')
OFFSETS = (-.1, .1, .5, 1.)


def main() -> None:
    """合否やP99を変えず、保留の映像根拠だけを保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path('logs/e22/HOLDS.json').read_text())
    for source in SOURCES:
        capture = cv2.VideoCapture(str(VIDEO_ROOT/f'{source}_first_0_900_20260925_v1.mp4'))
        for number, row in enumerate(audit['sources'][source]['rows']):
            tiles = []
            for offset in OFFSETS:
                stamp = row['held_sec']+offset
                capture.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
                ok, frame = capture.read()
                assert ok
                frame = cv2.resize(frame, (960, 540))
                x = 50 if row['side'] == '1P' else 555
                tile = cv2.copyMakeBorder(frame[70:515, x:x+350], 25, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(tile, f"{row['side']} {stamp:.3f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                tiles.append(tile)
            assert cv2.imwrite(str(OUT/f'{source}_{number}.jpg'), np.hstack(tiles))
        capture.release()


if __name__ == '__main__':
    main()
