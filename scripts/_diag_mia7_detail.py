"""証明前後・予告・実着弾の元画像を管理外へ抽出する。"""
from __future__ import annotations

import cv2
import numpy as np

from scripts._diag_mia7_extract import VIDEO, FRAMES
from scripts._diag_mia7_inspect import guard, save

DETAIL_TIMES = (687.4, 687.8, 688.1, 689.4, 690.533333, 690.566667,
                691.433333, 693.2, 694.6, 696.1, 697.3, 698.7, 699.0)
LANDING_STARTS = (699.7, 701.9)
STEP, COUNT = 0.1, 9
WIDTH, HEIGHT, COLS = 640, 360, 3


def main() -> None:
    """フレーム番号と読出し時刻を記録する。"""
    guard()
    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    times = list(DETAIL_TIMES)
    times += [start + offset * STEP for start in LANDING_STARTS for offset in range(COUNT)]
    manifest, thumbs = [], []
    for stamp in times:
        guard()
        index = round(stamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok
        name = f'detail_{stamp:.3f}.jpg'
        cv2.imwrite(str(FRAMES / name), frame)
        thumb = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.putText(thumb, f'{stamp:.3f}s', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        thumbs.append(thumb)
        manifest.append(dict(t=stamp, frame=index, path=str(FRAMES / name)))
    cap.release()
    for start in range(0, len(thumbs), COLS * COLS):
        sheet = np.zeros((HEIGHT * COLS, WIDTH * COLS, 3), np.uint8)
        for offset, thumb in enumerate(thumbs[start:start + COLS * COLS]):
            y, x = divmod(offset, COLS)
            sheet[y*HEIGHT:(y+1)*HEIGHT, x*WIDTH:(x+1)*WIDTH] = thumb
        cv2.imwrite(str(FRAMES / f'details_{start // (COLS * COLS)}.jpg'), sheet)
    save('detail_manifest.json', dict(video=str(VIDEO), fps=fps, frames=manifest))


if __name__ == '__main__':
    main()
