"""C上の既存動画を保存画像列として再投入する。仮想カメラは開かない。"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

VIDEO = 'C:/puyo1_tools/soak/media/video_d04.mp4'
DECODER = 'C:/puyo1_tools/soak/vcam_venv/Scripts/python.exe'
SHAPE = (1080, 1920, 3)
FRAME_BYTES = SHAPE[0] * SHAPE[1] * SHAPE[2]
FPS = 30
BELOW_NORMAL = 0x4000


class VideoFrames:
    """パイプの背圧でデコードを制限し、全画像をRAMに貯めない。"""

    def __init__(self, start: float, count: int) -> None:
        self.count = count
        self.process = subprocess.Popen([DECODER, str(Path(__file__).resolve()),
            str(start), str(count)], stdout=subprocess.PIPE, creationflags=BELOW_NORMAL)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Any:
        import numpy as np
        for _ in range(self.count):
            raw = self.process.stdout.read(FRAME_BYTES)
            if len(raw) != FRAME_BYTES:
                raise RuntimeError('保存動画のデコードが指定枚数より前に終了')
            yield np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE)

    def close(self) -> None:
        self.process.stdout.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


def decode() -> None:
    """専用デコーダのBGR画素を加工せず返す。"""
    import cv2
    import msvcrt
    cv2.setNumThreads(1)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    capture = cv2.VideoCapture(VIDEO)
    capture.set(cv2.CAP_PROP_POS_MSEC, float(sys.argv[1])*1000)
    try:
        for _ in range(int(sys.argv[2])):
            ok, image = capture.read()
            if not ok or image.shape != SHAPE:
                raise RuntimeError('動画の画素または解像度が不正')
            sys.stdout.buffer.write(image.tobytes())
    finally:
        capture.release()


if __name__ == '__main__':
    decode()
