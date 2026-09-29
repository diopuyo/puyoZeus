"""手番の実画像と既存ばたんきゅー検出を評価修正前に確認する。"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
from src.match_end_detector import MatchEndDetector

ROOT = Path("logs/e16")
VIDEO = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4")
STILLS = (2613.2166666666667, 2613.516666666667, 2613.616666666667, 2613.8166666666666,
          2699.483333333333, 2699.8166666666666)
START, END, STRIDE = 2698.5, 2701.1, 2
FRAME_SIZE = (1920, 1080)


def main() -> None:
    """原動画の同一フレームへ既存検出器を適用し、閾値は変更しない。"""
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    for stamp in STILLS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(stamp*fps))
        success, frame = capture.read()
        assert success
        cv2.imwrite(str(ROOT / f"source_{stamp:.3f}.png"), frame)
    detector = MatchEndDetector.load_default()
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(START*fps))
    results = []
    for index in range(int(START*fps), int(END*fps)):
        success, frame = capture.read()
        if not success:
            break
        if index % STRIDE:
            continue
        frame = cv2.resize(frame, FRAME_SIZE)
        value = detector.detect(frame)
        results.append(dict(t_sec=index/fps, detected=value.detected,
                            template=value.template_name, score=value.score))
    capture.release()
    (ROOT / "death_detection.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps([r for r in results if r["detected"]]), flush=True)


if __name__ == "__main__":
    main()
