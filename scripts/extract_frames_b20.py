"""B20: 動画の窓を stride2 で PNG 連番へ書き出す (Windows 配布版 Python は動画を開けないため)。
出力名は frame_<元frame番号>.png。本番の VideoFileSource と同じ frame 列 (start, start+2, ...)。"""
import sys
from pathlib import Path

import cv2

video, start_sec, frames, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
first = int(start_sec*fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, first)
for i in range(frames):
    ok, image = cap.read()
    cap.grab()
    if not ok:
        break
    cv2.imwrite(str(out/f'frame_{first+2*i:07d}.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
print('extracted', i+1, 'fps', fps)
