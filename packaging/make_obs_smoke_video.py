"""前回の固定画素から、FFmpeg不要の動画入力スモーク用MJPEGを作る。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equiv_frames import CachedCapture, META_NAME

START_SEC = 225
END_SEC = 270
FRAME_SIZE = (1920, 1080)


def main() -> None:
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    target = options.output.resolve()
    local = Path('D:/puyo_analyzer/local_frames').resolve()
    if not target.is_relative_to(local) or target.exists():
        raise ValueError('未作成のPC1 local_frames内だけに出力できます')
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((options.cache / META_NAME).read_text(encoding='utf-8'))
    fps = metadata['fps']
    capture = CachedCapture(options.cache, target.with_suffix('.pixels.json'))
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(START_SEC * fps))
    writer = cv2.VideoWriter(str(target), cv2.CAP_OPENCV_MJPEG,
                            cv2.VideoWriter_fourcc(*'MJPG'), fps, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError('MJPEG writerを開けません')
    try:
        for _ in range(round((END_SEC - START_SEC) * fps)):
            ok, frame = capture.read()
            if not ok:
                raise EOFError('固定画素が不足しています')
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    check = cv2.VideoCapture(str(target))
    assert check.isOpened() and check.read()[0]
    print(json.dumps(dict(video=str(target), fps=fps, backend=check.getBackendName(),
                          frames=check.get(cv2.CAP_PROP_FRAME_COUNT))))
    check.release()


if __name__ == '__main__':
    main()
