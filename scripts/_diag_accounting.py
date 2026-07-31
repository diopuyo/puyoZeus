"""game B の会計イベント(finalize/offset/cap)を INFO ログで採取し精度を診断。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402

VIDEO = "data/frames/video_29.mp4"
START, END = 145.0, 283.0  # game A頭から game B終わりまで(会計を正しく初期化)


def main() -> None:
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id("video_29")
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START * fps))
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    for fi in range(int(START * fps), int(END * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080))
        t = fi / fps
        r = pipe.update(fi, t, f)
        _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t,
                     tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
    cap.release()


if __name__ == "__main__":
    main()
