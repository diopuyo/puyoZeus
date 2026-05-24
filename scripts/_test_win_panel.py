"""WinPanelDetector の簡易動作確認スクリプト。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.win_panel import WinPanelDetector


def main() -> None:
    det = WinPanelDetector.load_default()
    test_paths = [
        # video_02 試合中フレーム
        "data/frames/review_video_02/frame_0210s.png",
        "data/frames/review_video_02/frame_0225s.png",
        "data/frames/review_video_02/frame_0240s.png",
        "data/frames/review_video_02/frame_0270s.png",
        "data/frames/review_video_02/frame_0285s.png",
        "data/frames/review_video_02/frame_0315s.png",
        # 非試合
        "tests/fixtures/frames/not_match_browser_0050.png",
        "tests/fixtures/frames/not_match_vs_0170.png",
        # video_01 サンプル
        "data/frames/sample/frame_0300s.png",
        "data/frames/sample/frame_0600s.png",
        "data/frames/sample/frame_2100s.png",
        # 境界判定の after/before 数枚
        "data/verify/match_boundaries/video_02/end_02985s_before.png",
        "data/verify/match_boundaries/video_02/start_02949s_after.png",
        "data/verify/match_boundaries/video_02/end_02364s_after.png",
    ]
    for p in test_paths:
        path = Path(p)
        if not path.exists():
            print(f"SKIP: {p}")
            continue
        f = cv2.imread(str(path))
        r = det.detect(f)
        tag = "PANEL" if r.present else "NO   "
        print(f"{tag}  score={r.score:.3f}  {p}")


if __name__ == "__main__":
    main()
