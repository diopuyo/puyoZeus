"""画素入力を逐次再生して盤面認識の段別費用・重複を測る診断台。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
from collections import Counter
import ctypes
import functools
import hashlib
import inspect
import json
import os
import pickle
from pathlib import Path
import sys
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MS = 1000.0
WARMUP = 60
MIN_FREE = 6 * 1024**3
SOURCE_FPS = 30.0
SOURCE_SHAPE = (1080, 1920, 3)
ROWS: list[dict[str, Any]] = []
STACK: list[list[Any]] = []
STAGES: Counter = Counter()
COUNTS: Counter = Counter()
PATCHES: set[bytes] = set()
MINIMUM_FREE = float('inf')


def system_times() -> tuple[int, int]:
    """Windows全体のCPU時間を取得し、他レーン込みの負荷を残す。"""
    from ctypes import wintypes
    idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    values = [(f.dwHighDateTime << 32) | f.dwLowDateTime for f in (idle, kernel, user)]
    return values[0], values[1] + values[2]


def wrap(owner: Any, name: str, label: str) -> None:
    """包含時間と子呼出を除く自己時間を両方記録する。"""
    original = getattr(owner, name)

    @functools.wraps(original)
    def measured(*args: Any, **kwargs: Any) -> Any:
        entry = [time.perf_counter(), 0.0]
        STACK.append(entry)
        COUNTS[label] += 1
        if label == 'cnn_batch':
            for patch in args[1]:
                key = (str(patch.shape) + patch.dtype.str).encode() + patch.tobytes()
                COUNTS['patches'] += 1
                COUNTS['duplicate_patches'] += key in PATCHES
                PATCHES.add(key)
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - entry[0]
            STACK.pop()
            STAGES[label] += elapsed * MS
            STAGES[label + '_self'] += (elapsed - entry[1]) * MS
            if STACK:
                STACK[-1][1] += elapsed
    descriptor = inspect.getattr_static(owner, name)
    setattr(owner, name, staticmethod(measured) if isinstance(descriptor, staticmethod) else measured)


def instrument() -> None:
    from src.recognition_pipeline import RecognitionPipeline
    from src.image_reader import ImageReader, ColorClassifier, BoardRegion
    from src.hybrid_classifier import HybridClassifier
    from src.patch_classifier import CnnPatchClassifier
    from src import cnn_onnx
    from src import board_rules
    targets = [(RecognitionPipeline, 'update', 'pipeline'),
               (ImageReader, 'read_both_boards', 'boards'),
               (ImageReader, 'read_board', 'board'),
               (ImageReader, '_board_hsv_frame', 'board_hsv'),
               (BoardRegion, 'cell_sample_rect', 'patch_rect'),
               (ImageReader, '_is_empty_tier1', 'background_gate'),
               (ImageReader, '_apply_profile_filter', 'profile_filter'),
               (ImageReader, '_infer_hidden_rows', 'hidden_rows'),
               (board_rules, 'clear_floating_above_gap', 'floating_cleanup'),
               (HybridClassifier, 'classify_batch', 'hybrid'),
               (ColorClassifier, 'classify', 'hsv_classify'),
               (CnnPatchClassifier, 'predict_proba_batch', 'cnn_batch'),
               (CnnPatchClassifier, '_patch_to_tensor', 'preprocess'),
               (CnnPatchClassifier, '_forward', 'forward'),
               (cnn_onnx, 'run_logits', 'onnx')]
    for target, method, label in targets:
        wrap(target, method, label)


def check_resources() -> None:
    """計測開始前と各コマで共同作業の保留・メモリ下限を守る。"""
    if Path(_public_dev_text('<work>/HOLD_FOR_OBS')).exists():
        raise RuntimeError('HOLD_FOR_OBS があるため停止')
    status = ctypes.create_string_buffer(64)
    ctypes.cast(status, ctypes.POINTER(ctypes.c_ulong))[0] = 64
    ctypes.windll.kernel32.GlobalMemoryStatusEx(status)
    free = ctypes.c_ulonglong.from_buffer(status, 16).value
    global MINIMUM_FREE
    MINIMUM_FREE = min(MINIMUM_FREE, free)
    if free < MIN_FREE:
        raise RuntimeError('空きメモリ6GiB未満')


def run(args: argparse.Namespace) -> None:
    import numpy as np
    from src.recognition_pipeline import RecognitionPipeline
    config = json.loads(args.config.read_text(encoding='utf-8'))
    pipe = RecognitionPipeline.load_default(**config)
    load_before = system_times()
    shape = SOURCE_SHAPE
    frame_bytes = int(np.prod(shape))
    capture = subprocess.Popen(['C:/puyo1_tools/soak/vcam_venv/Scripts/python.exe',
        '-B', str(Path(__file__)), '--decode', str(args.video), str(args.start),
        str(args.frames)], stdout=subprocess.PIPE, creationflags=0x4000)
    try:
        for index in range(args.frames):
            check_resources()
            raw = capture.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            image = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            STAGES.clear(); COUNTS.clear(); PATCHES.clear()
            started = time.perf_counter()
            result = pipe.update(index, args.start + index / args.fps, image)
            elapsed = (time.perf_counter() - started) * MS
            ROWS.append(dict(index=index, stages=dict(STAGES), counts=dict(COUNTS),
                states=[result.p1.state.name, result.p2.state.name], ms=elapsed,
                digest=hashlib.sha256(pickle.dumps(result)).hexdigest(),
                raw=[s.cnn_board._grid.tolist() for s in (result.p1, result.p2)],
                confirmed=[None if s.confirmed_board is None else s.confirmed_board._grid.tolist()
                           for s in (result.p1, result.p2)]))
    finally:
        capture.terminate()
        capture.wait()
        if not ROWS:
            raise RuntimeError('入力画像0枚: 計測不成立')
        args.output.write_text(json.dumps(ROWS), encoding='utf-8')
        cache = getattr(pipe, '_live_delta_cache', None)
        receipt = dict(min_free_gib=MINIMUM_FREE / 1024**3, mode=args.mode,
                       frames=len(ROWS), source=str(args.video), start=args.start, fps=args.fps)
        load_after = system_times()
        receipt['system_cpu_percent'] = 100 * (1 - (load_after[0] - load_before[0]) /
                                              (load_after[1] - load_before[1]))
        if cache:
            receipt['cache'] = {k: getattr(cache, k) for k in
                ('requested', 'inferred', 'reused', 'full_reads')}
        args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=Path('C:/puyo1_tools/soak/media/video_d04.mp4'))
    parser.add_argument('--config', type=Path, default=Path('C:/puyo1_tools/soak/run_k/output/recognition_config.json'))
    parser.add_argument('--start', type=float, default=650.0)
    parser.add_argument('--fps', type=float, default=SOURCE_FPS)
    parser.add_argument('--frames', type=int, default=900)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('off', 'dedup', 'delta'), default='off')
    parser.add_argument('--app', type=Path, default=ROOT)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.app))
    check_resources()
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    os.environ.update(PUYO_CNN_BACKEND='onnx', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PUYO_FAST_TELOP='1', PUYO_SCORE_ZERO_FROM_OCR='1',
        PUYO_LIVE_CNN_DEDUP=str(int(args.mode == 'dedup')),
        PUYO_LIVE_DELTA_CELLS=str(int(args.mode == 'delta')))
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(1)
    if args.profile:
        instrument()
    run(args)


def decode() -> None:
    """既存仮想カメラ台の動画デコーダだけを使い、画像を無圧縮で渡す。"""
    import cv2
    import msvcrt
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(sys.argv[2])
    capture.set(cv2.CAP_PROP_POS_MSEC, float(sys.argv[3]) * MS)
    try:
        for _ in range(int(sys.argv[4])):
            ok, image = capture.read()
            if not ok:
                break
            if image.shape != SOURCE_SHAPE:
                image = cv2.resize(image, (SOURCE_SHAPE[1], SOURCE_SHAPE[0]), interpolation=cv2.INTER_AREA)
            sys.stdout.buffer.write(image.tobytes())
    finally:
        capture.release()


if __name__ == '__main__':
    decode() if '--decode' in sys.argv else main()
