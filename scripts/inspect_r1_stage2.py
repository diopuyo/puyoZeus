"""段2の固定対象セルを原画像と独立CNN/HSVで照合する。"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from scripts._d3_observe import readers, observe, ZENCHI
from scripts.enrich_e26_midchain import frame_at
from src.image_reader import DEFAULT_P2_REGION

OUT = Path('logs/r1/stage2')
CELLS = ((1, 3), (1, 4), (1, 5), (2, 5))
FIRST, LAST, FPS = 2749.7, 2751.5, 60
SCENES = (2750.283333, 2750.300000, 2750.816667, 2750.833333)


def main() -> None:
    """対象の前後を全原フレームで保存し、合図時点を並べて表示する。"""
    classifiers = readers()
    cap = cv2.VideoCapture(str(ZENCHI), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    rows, tiles = [], []
    indices = {round(t*FPS) for t in SCENES}
    for index in range(round(FIRST*FPS), round(LAST*FPS)+1):
        frame = frame_at(cap, index)
        values = observe(frame, {1}, *classifiers)['1']
        row = dict(frame=index, t=index/FPS, **values)
        row['target_cnn'] = [values['cnn'][(r-1)*6+c] for r, c in CELLS]
        row['target_hsv'] = [values['hsv'][(r-1)*6+c] for r, c in CELLS]
        rows.append(row)
        if index in indices:
            region = DEFAULT_P2_REGION
            tile = frame[region.y:region.y+region.height, region.x:region.x+region.width].copy()
            tile = cv2.copyMakeBorder(tile, 45, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, f'{index/FPS:.6f}', (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
            tiles.append(tile)
            cv2.imwrite(str(OUT/f'frame_{index}.png'), frame)
    cap.release()
    (OUT/'independent_observations.json').write_text(json.dumps(rows))
    cv2.imwrite(str(OUT/'signal_frames.jpg'), np.hstack(tiles))


if __name__ == '__main__':
    main()
