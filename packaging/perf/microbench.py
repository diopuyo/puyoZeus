"""OS 間・実行環境間で、認識が使う基本演算の 1 回あたり時間 (µs) を並べる。1 スレッド固定。
使い方: python microbench.py [出力 json]。各演算は暖機後に REPEAT 回の最小/中央値を取る。"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(1)
REPEAT, WARM = 200, 20
MICRO = 1e6
FRAME = (1080, 1920, 3)
rng = np.random.default_rng(0)
frame = rng.integers(0, 255, FRAME, dtype=np.uint8)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
roi = np.ascontiguousarray(gray[300:700, 500:1400])
template = np.ascontiguousarray(roi[100:160, 200:400])
patches = torch.from_numpy(rng.random((144, 3, 24, 24), dtype=np.float32))
conv = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, padding=1),
                           torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)).eval()
small = rng.random((14, 14), dtype=np.float32)


def python_loop() -> None:
    total = 0
    for value in range(20000):
        total += value*value
    return None


def many_small_numpy() -> None:
    for _ in range(200):
        small.mean(); small.astype(np.float64); np.partition(small.ravel(), 5)


CASES: dict[str, Callable[[], object]] = {
    'python_loop_20k': python_loop,
    'small_numpy_x200': many_small_numpy,
    'zeros_like_1080p': lambda: np.zeros_like(frame),
    'frame_copy_1080p': lambda: frame.copy(),
    'cvtColor_bgr2gray_1080p': lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    'cvtColor_bgr2hsv_1080p': lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
    'resize_1080p_half_area': lambda: cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA),
    'matchTemplate_ccoeff_normed': lambda: cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED),
    'dft_roi': lambda: cv2.dft(roi.astype(np.float32)),
    'integral2_roi': lambda: cv2.integral2(roi),
    'torch_conv_144patches': lambda: conv(patches),
}


def measure(function: Callable[[], object]) -> dict[str, float]:
    for _ in range(WARM):
        function()
    samples = []
    for _ in range(REPEAT):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter()-start)*MICRO)
    samples.sort()
    return dict(min=samples[0], p50=samples[len(samples)//2])


def main() -> None:
    with torch.no_grad():
        result = {name: measure(function) for name, function in CASES.items()}
    text = json.dumps(dict(platform=sys.platform, python=sys.version.split()[0], cv2=cv2.__version__,
                           numpy=np.__version__, torch=torch.__version__, results=result), indent=1)
    if len(sys.argv) > 1:
        open(sys.argv[1], 'w', encoding='utf-8').write(text)
    print(text)


if __name__ == '__main__':
    main()
