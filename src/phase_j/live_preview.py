"""入力の縮小画像をメモリだけに保持する。配布アプリの子だけで有効。"""
from __future__ import annotations

import os
import atexit
from multiprocessing.shared_memory import SharedMemory
import struct
import threading
import time
from typing import Any

PREVIEW_SIZE = (320, 180)
PREVIEW_INTERVAL = 1.0
JPEG_QUALITY = 65
PREVIEW_CAPACITY = 128 * 1024
HEADER = struct.Struct('<QI')
SEQUENCE = struct.Struct('<Q')
LENGTH = struct.Struct('<I')
PREVIEW_MEMORY_ENV = 'PUYO_INPUT_PREVIEW_MEMORY'
_lock = threading.Lock()
_image = b''
_updated = 0.0
_memory: SharedMemory | None = None


def shared_memory() -> SharedMemory | None:
    global _memory
    name = os.environ.get(PREVIEW_MEMORY_ENV)
    if not name:
        return None
    if _memory is None or _memory.name != name:
        if _memory is not None:
            _memory.close()
        _memory = SharedMemory(name=name)
        atexit.register(_memory.close)
    return _memory


def publish_image(image: bytes) -> None:
    memory = shared_memory()
    if memory is None or len(image) > PREVIEW_CAPACITY - HEADER.size:
        return
    sequence = HEADER.unpack_from(memory.buf)[0]
    HEADER.pack_into(memory.buf, 0, sequence + 1, 0)
    memory.buf[HEADER.size:HEADER.size + len(image)] = image
    LENGTH.pack_into(memory.buf, SEQUENCE.size, len(image))
    SEQUENCE.pack_into(memory.buf, 0, sequence + 2)


def shared_image(memory: SharedMemory) -> bytes:
    """更新中や途中で入れ替わった画像は次の表示周期へ送る。"""
    sequence, size = HEADER.unpack_from(memory.buf)
    if sequence % 2 or not 0 < size <= PREVIEW_CAPACITY - HEADER.size:
        return b''
    image = bytes(memory.buf[HEADER.size:HEADER.size + size])
    return image if HEADER.unpack_from(memory.buf)[0] == sequence else b''


def thumbnail(frame: Any) -> bytes:
    import cv2
    if frame is None or not frame.size:
        return b''
    small = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return encoded.tobytes() if ok else b''


def offer_frame(frame: Any) -> None:
    global _image, _updated
    if os.environ.get('PUYO_INPUT_PREVIEW') != '1':
        return
    now = time.monotonic()
    with _lock:
        if now - _updated < PREVIEW_INTERVAL:
            return
        _image = thumbnail(frame)
        publish_image(_image)
        _updated = now


def current_image() -> bytes:
    with _lock:
        memory = shared_memory()
        if memory is not None:
            return shared_image(memory)
        return _image
