"""cv2 を 1 スレッドに固定して visualize_advantage_overlay.main を呼ぶ非侵襲ラッパー。

scripts/_collect_1t.py と同じ方針 (matchTemplate はスレッド並列が効かず、
多プロセス並列時のスレッドプール競合を避けるため 1 プロセス=1コアに固定)。
ザッピングレビュー動画生成 (2026-07-23) の並列レンダー用。src/ 側は無改修。
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2.setNumThreads(1)

from scripts.visualize_advantage_overlay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
