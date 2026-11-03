"""等価比較専用。動画を一度だけ復号し、全環境で同じBGR画素を読む。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

META_NAME = 'frames.json'
ARRAY_NAME = 'frames.bgr'
PROGRESS_INTERVAL = 300
MILLISECONDS = 1000.0


def digest(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def create_cache(video: Path, directory: Path, start: float, end: float) -> None:
    """絶対フレーム番号を保持し、ウォームアップを含む連続フレームを保存する。"""
    directory.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f'動画を開けません: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    first, last = round(start * fps), round(end * fps)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    hashes: list[str] = []
    shape: tuple[int, ...] | None = None
    stream = (directory / ARRAY_NAME).open('wb')
    try:
        for index in range(first, last):
            ok, frame = capture.read()
            if not ok:
                raise EOFError(f'復号中に入力終了: {index}')
            shape = frame.shape
            stream.write(frame.tobytes())
            hashes.append(digest(frame))
            if len(hashes) % PROGRESS_INTERVAL == 0:
                print(f'保存: {len(hashes)}/{last-first}', flush=True)
        stream.close()
        metadata = dict(source=video.name, fps=fps, first=first, last=last, total=total,
                        shape=list(shape), sha256=hashes, opencv=cv2.__version__,
                        decoder_build=cv2.getBuildInformation())
        (directory / META_NAME).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    finally:
        stream.close()
        capture.release()


class CachedCapture:
    """復号器を使わない検査専用VideoCapture。各readで保存時ハッシュを照合する。"""

    def __init__(self, directory: Path, audit: Path) -> None:
        self.meta = json.loads((directory / META_NAME).read_text(encoding='utf-8'))
        self.stream = (directory / ARRAY_NAME).open('rb')
        self.frame_bytes = int(np.prod(self.meta['shape']))
        expected_bytes = len(self.meta['sha256']) * self.frame_bytes
        if (directory / ARRAY_NAME).stat().st_size != expected_bytes:
            self.stream.close()
            raise ValueError('固定フレーム列のサイズが不正です（複製未完了または破損）')
        self.position = 0
        self.opened = True
        self.audit = audit
        self.seen: list[dict[str, Any]] = []

    def isOpened(self) -> bool:
        return self.opened

    def getBackendName(self) -> str:
        return 'FIXED_BGR_CACHE'

    def get(self, key: int) -> float:
        values = {cv2.CAP_PROP_FPS: self.meta['fps'], cv2.CAP_PROP_FRAME_COUNT: self.meta['total'],
                  cv2.CAP_PROP_FRAME_WIDTH: self.meta['shape'][1],
                  cv2.CAP_PROP_FRAME_HEIGHT: self.meta['shape'][0],
                  cv2.CAP_PROP_POS_FRAMES: self.position,
                  cv2.CAP_PROP_POS_MSEC: self.position / self.meta['fps'] * MILLISECONDS}
        if key not in values:
            raise ValueError(f'未対応のcapture.get: {key}')
        return float(values[key])

    def set(self, key: int, value: float) -> bool:
        if key == cv2.CAP_PROP_POS_FRAMES:
            self.position = round(value)
        elif key == cv2.CAP_PROP_POS_MSEC:
            self.position = round(value / MILLISECONDS * self.meta['fps'])
        else:
            raise ValueError(f'未対応のcapture.set: {key}')
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        offset = self.position - self.meta['first']
        if not 0 <= offset < len(self.meta['sha256']):
            raise IndexError(f'保存フレーム範囲外の読込: {self.position}')
        self.stream.seek(offset * self.frame_bytes)
        frame = np.frombuffer(self.stream.read(self.frame_bytes), dtype=np.uint8).reshape(self.meta['shape']).copy()
        hashed = digest(frame)
        if hashed != self.meta['sha256'][offset]:
            raise ValueError(f'入力画素ハッシュ不一致: {self.position}')
        self.seen.append(dict(frame=self.position, sha256=hashed))
        self.position += 1
        return True, frame

    def grab(self) -> bool:
        self.read()
        return True

    def release(self) -> None:
        if self.opened and self.seen:
            self.audit.parent.mkdir(parents=True, exist_ok=True)
            self.audit.write_text(json.dumps(self.seen), encoding='utf-8')
        self.opened = False
        self.stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    args = parser.parse_args()
    create_cache(args.video, args.directory, args.start, args.end)


if __name__ == '__main__':
    main()
