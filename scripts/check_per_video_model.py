"""per_video_model_selector の動作確認。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.per_video_model_selector import (
    V17B_BEST_VIDEOS, select_model_for_video,
)


def main() -> int:
    for vid in range(1, 20):
        if vid == 18:
            continue
        path = f"data/frames/video_{vid:02d}.mp4"
        model = select_model_for_video(path)
        print(f"v{vid:02d}: {model}")
    print(f"\nV17B_BEST_VIDEOS: {sorted(V17B_BEST_VIDEOS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
