"""t=136-144 両サイド精密トレース (scratch): 2Pの相殺漏れ確認用.

各フレームで 1P/2P の 状態・生score・平滑score・forecast を並べ、
2Pが1P連鎖中に小連鎖(~280点)を撃って finalize/相殺されているかを見る。
INFOログで finalize[p2]/offset も出す。
"""
from __future__ import annotations

import logging
import sys

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, ".")

from src.ojama_accounting import OjamaAccountingTracker
from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama

VIDEO = "data/frames/video_124_4min.mp4"
TARGET_W, TARGET_H = 1920, 1080
T0, T1 = 136.0, 144.0


def main() -> None:
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if hasattr(pipe, "set_video_id"):
        pipe.set_video_id("video_124")
    tracker = OjamaAccountingTracker(); tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    prev1, prev2 = BoardState.MENU, BoardState.MENU
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ps1 = ps2 = None
    print("  t     st1        st2        sc1   sc2   fc1 fc2")
    for fi in range(int((T1 + 1) * fps)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        snap = _drive_ojama(tracker, r.p1, r.p2, prev1, prev2, t, tp1, tp2, pipe)
        prev1, prev2 = r.p1.state, r.p2.state
        if T0 <= t <= T1:
            s1, s2 = r.p1.score, r.p2.score
            # 2Pのscore変化 or 状態がCHAIN系のフレームだけ出す(冗長回避)
            interesting = (s2 != ps2 or s1 != ps1
                           or r.p2.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE))
            if interesting:
                print(f"{t:6.2f} {str(r.p1.state)[11:]:10s} {str(r.p2.state)[11:]:10s} "
                      f"{str(s1):>5s} {str(s2):>5s}  {snap.forecast_p1:3d} {snap.forecast_p2:3d}")
            ps1, ps2 = s1, s2
    cap.release()


if __name__ == "__main__":
    main()
