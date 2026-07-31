# cv2スレッド数を変えて collect 20秒窓の所要時間を比較(過剰割当が競合原因か検証)
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if n_threads > 0:
    cv2.setNumThreads(n_threads)
print(f"cv2.getNumThreads()={cv2.getNumThreads()} (要求={n_threads})")

from scripts.collect_indicators_v2 import collect

t0 = time.perf_counter()
n = collect(
    Path("/home/ryouj/frames/video_33.mp4"),
    Path(f"/tmp/cv2test_{n_threads}.csv"),
    max_sec=15.0,
    start_sec=1200.0,
    board_npz_path=None,
)
dt = time.perf_counter() - t0
print(f"threads={n_threads} rows={n} wall={dt:.1f}s  fps={15*30/dt:.2f}")
