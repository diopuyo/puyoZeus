"""既知の早期確定場面でNEXTの実移動を画素から確認する。"""
from __future__ import annotations
import json
import os
import cv2
import numpy as np
from scripts._d3_inventory import OUT
from scripts._d3_observe import ZENCHI
from scripts.enrich_e26_midchain import frame_at

START, END, FPS = 2749.6, 2751.0, 60
PATCH_HALF = 16
SEARCH_X = 12
SEARCH_Y = 48
MIN_NCC = .80
MIN_GAIN = .08
MIN_SHIFT = 2


def next_translation(previous: np.ndarray, current: np.ndarray, side: int) -> dict:
    """NEXT内部の絵柄の平行移動を、輝度変化から区別する。"""
    centers = ((747, 199), (1172, 199))
    cx, cy = centers[side]
    patch = previous[cy-PATCH_HALF:cy+PATCH_HALF, cx-PATCH_HALF:cx+PATCH_HALF]
    search = current[cy-PATCH_HALF-SEARCH_Y:cy+PATCH_HALF+SEARCH_Y,
                     cx-PATCH_HALF-SEARCH_X:cx+PATCH_HALF+SEARCH_X]
    scores = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, high, _, location = cv2.minMaxLoc(scores)
    dx, dy = location[0]-SEARCH_X, location[1]-SEARCH_Y
    gain = high-float(scores[SEARCH_Y, SEARCH_X])
    # 同色組では、上へ動いた下側ぷよへの対応によりdyが正になる場合もある。
    return dict(moving=high >= MIN_NCC and gain >= MIN_GAIN and abs(dy) >= MIN_SHIFT,
                ncc=high, gain=gain, dx=dx, dy=dy)


def main() -> None:
    """操作中組ぷよを盤面に書いた前後のNEXT欄を保存する。"""
    os.nice(19)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(ZENCHI), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    samples, rows, previous, previous_full = [], [], None, None
    for index in range(round(START*FPS), round(END*FPS)):
        frame = frame_at(cap, index)
        crop = frame[150:400, 1095:1220]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            moving = np.linalg.norm(flow, axis=2) > 2
            dy = flow[:, :, 1][moving]
            rows.append(dict(t=index/FPS, translation=next_translation(previous_full, full, 1), fraction=float(moving.mean()),
                dy=float(np.median(dy)) if len(dy) else 0,
                diff=float(np.abs(gray.astype(float)-previous).mean())))
        previous = gray
        previous_full = full
        if index % 3 == 0:
            tile = np.zeros((280, 125, 3), dtype=np.uint8)
            tile[30:] = crop
            cv2.putText(tile, f'{index/FPS:.2f}', (1,20), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            samples.append(tile)
    cap.release()
    sheets = [np.hstack(samples[n:n+10]) for n in range(0, len(samples), 10)]
    for n, sheet in enumerate(sheets):
        cv2.imwrite(str(OUT/f'next_scene_{n}.jpg'), sheet)
    (OUT/'next_scene_motion.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps([r for r in rows if r['translation']['moving']]))


if __name__ == '__main__':
    main()
