"""game B t=47前後の 3成分(モデル/圧力/threat)と最終ブレンドを分解表示。"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    _train_model, _score_advantage, PressureTracker, _threat,
    W_PRESSURE, W_MODEL, W_THREAT,
)

VIDEO = "data/frames/video_29.mp4"
START, END, WARM = 202.0, 262.0, 16.0


def main() -> None:
    model = _train_model("video_29")
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id("video_29")
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((START - WARM) * fps))
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    pt = PressureTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    step = int(0.5 * fps)
    print("t_rel | model  圧力  threat | ブレンド | pot火力1P/2P")
    for fi in range(int((START - WARM) * fps), int(END * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080))
        t = fi / fps
        r = pipe.update(fi, t, f)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if b1 is None or b2 is None:
            continue
        m, _, _ = _score_advantage(model, b1, b2, snap)
        pres = pt.update(iv.board_ojama_count(b1).raw, iv.board_ojama_count(b2).raw)
        el = tr._elapsed(t)
        thr = _threat(b1, b2, el)
        blend = W_PRESSURE * pres + W_MODEL * m + W_THREAT * thr
        if fi % step or t < START + 44 or t > START + 60:
            continue
        p1p = iv.potential_fire_power(b1, el).raw
        p2p = iv.potential_fire_power(b2, el).raw
        print(f"{t - START:5.1f} | {m:+5.0f} {pres:+5.0f} {thr:+6.0f} | "
              f"{blend:+7.0f} | {p1p:4.0f}/{p2p:4.0f}")
    cap.release()


if __name__ == "__main__":
    main()
