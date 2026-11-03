"""再生中に発見した追加時刻の原画像を同じプロセスで保存する。"""
from __future__ import annotations

import json

import cv2

from scripts._diag_set2_falsedeath_extract import OUT, VIDEO, OFFSETS

EXTRA = (('game15_later', 4620.7), ('game58_write', 7002.516667),
         ('game45_write', 6308.6), ('game13_write', 4417.366667), ('game45_later', 6379.2))


def main() -> None:
    """初回だけを検査して後続の誤確定を見逃さないための追加証拠。"""
    path = OUT / 'extra_frames.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    todo = [(name, stamp) for name, stamp in EXTRA if not (OUT / f'{name}_+1.0.jpg').exists()]
    if not todo:
        return
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    for name, stamp in todo:
        for offset in OFFSETS:
            index = round((stamp + offset) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok
            file = f'{name}_{offset:+.1f}.jpg'
            assert cv2.imwrite(str(OUT / file), frame)
            rows.append(dict(file=file, frame=index, t_sec=index / fps))
    cap.release()
    path.write_text(json.dumps(rows, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
