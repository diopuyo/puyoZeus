"""P1確定診断: forecast固定/残留がバグか正常かを1回で判定 (scratch).

現行の collect ドライバ (_drive_ojama = tsumo_count増分drain適用済) を忠実に再現し、
注目区間 (t=50〜65s の相殺+forecast_p2固定, t>=168s の末尾forecast_p1残留) を
1フレームずつ出力する。

見るべき点:
- forecast_p2=26 の区間で 2P の tsumo_count が増える(=着地あり)のに fc_p2 が減らない → バグ
  減る → 修正で解消 / そもそも着地が無い → 正常(連鎖中は降らない)
- 末尾 fc_p1 が高いまま: 着地があるのに減らない → バグ / 着地無し(試合終了/埋没) → 正常
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, ".")

from src.board_state_machine import BoardState
from src.ojama_accounting import OjamaAccountingTracker
from src.recognition_pipeline import RecognitionPipeline
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama

VIDEO = "data/frames/video_124_4min.mp4"
TARGET_W, TARGET_H = 1920, 1080
WINDOWS = [(131.0, 137.0), (140.0, 146.0)]


def _in_window(t: float) -> bool:
    return any(a <= t <= b for a, b in WINDOWS)


def main() -> None:
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id("video_124")
    tracker = OjamaAccountingTracker()
    tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    prev1, prev2 = BoardState.MENU, BoardState.MENU

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(148 * fps)
    ptc1 = ptc2 = 0
    pf1 = pf2 = 0
    print(f"fps={fps:.2f}")
    print("  t    st1         st2         tc1(+d) tc2(+d)  fc1  fc2  drain")
    for fi in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        result = pipeline.update(fi, t, frame)
        snap = _drive_ojama(
            tracker, result.p1, result.p2, prev1, prev2, t, tp1, tp2, pipeline,
        )
        prev1, prev2 = result.p1.state, result.p2.state
        tc1 = pipeline.tsumo_count("1P")
        tc2 = pipeline.tsumo_count("2P")
        d1, d2 = tc1 - ptc1, tc2 - ptc2
        f1, f2 = snap.forecast_p1, snap.forecast_p2
        if _in_window(t) and (d1 or d2 or f1 != pf1 or f2 != pf2):
            drain = ""
            if d1 > 0:
                drain += f"1P着地x{d1} "
            if d2 > 0:
                drain += f"2P着地x{d2} "
            print(
                f"{t:6.2f} {str(prev1):11s} {str(prev2):11s} "
                f"{tc1:3d}(+{d1}) {tc2:3d}(+{d2})  {f1:3d}  {f2:3d}  {drain}"
            )
        ptc1, ptc2, pf1, pf2 = tc1, tc2, f1, f2
    cap.release()
    print(f"\n最終: fc_p1={pf1} fc_p2={pf2}  tsumo_count 1P={ptc1} 2P={ptc2}")


if __name__ == "__main__":
    main()
