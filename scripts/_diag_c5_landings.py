"""実着弾の開始区間を100ms間隔の映像で確認する。"""
from __future__ import annotations

import cv2
import numpy as np
from scripts._diag_c5_extract import VIDEO
from scripts._diag_c5_inspect import OUT, guard, save

TIMES = [1731.5+i/10 for i in range(10)] + [1733.4+i/10 for i in range(10)]


def main() -> None:
    """盤面と予告を同時に残す。"""
    guard()
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    thumbnails = []
    for stamp in TIMES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(stamp*fps))
        ok, frame = capture.read()
        assert ok
        cv2.imwrite(str(OUT / f'landing_{stamp:.3f}.jpg'), frame)
        crop = cv2.resize(frame[70:950, 240:700], (230,440))
        cv2.putText(crop, f'{stamp:.3f}', (5,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
        thumbnails.append(crop)
    capture.release()
    for start in (0,10):
        sheet = np.concatenate([np.concatenate(thumbnails[i:i+5], axis=1) for i in (start,start+5)])
        cv2.imwrite(str(OUT / f'landing_sheet_{start//10}.jpg'), sheet)
    save('landing_frame_times.json', TIMES)


if __name__ == '__main__':
    main()
