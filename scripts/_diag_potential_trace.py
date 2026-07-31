"""Phase1: game B 終盤で SAKI/あん の potential_fire_power が
どう伸びるかを dense trace。あん128→816 の伸び方を見て真因切り分け。"""
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

VIDEO = "data/frames/video_29.mp4"
START, END, WARM = 202.0, 276.0, 16.0  # game B終盤まで


def main() -> None:
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id("video_29")
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((START - WARM) * fps))
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    step = int(0.5 * fps)
    print("t_rel| SAKI: 色/邪/potential  reach | あん: 色/邪/potential reach | st1/st2 sc1/sc2")
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
        _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t,
                     tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if fi % step or t < START + 45 or b1 is None or b2 is None:
            continue
        el = tr._elapsed(t)
        c1 = int(iv.board_color_puyo_total(b1).raw); o1 = int(iv.board_ojama_count(b1).raw)
        c2 = int(iv.board_color_puyo_total(b2).raw); o2 = int(iv.board_ojama_count(b2).raw)
        p1 = iv.potential_fire_power(b1, el).raw
        p2 = iv.potential_fire_power(b2, el).raw
        rc1 = iv.reach_fire_power(b1, r.p1.next_pair, r.p1.dnext_pair, el).value.raw
        rc2 = iv.reach_fire_power(b2, r.p2.next_pair, r.p2.dnext_pair, el).value.raw
        print(f"{t - START:5.1f}| {c1:2d}/{o1:2d}/{p1:4.0f} {rc1:4.0f} "
              f"| {c2:2d}/{o2:2d}/{p2:4.0f} {rc2:4.0f} "
              f"| {str(r.p1.state)[11:14]}/{str(r.p2.state)[11:14]} "
              f"{r.p1.score}/{r.p2.score}")
    cap.release()


if __name__ == "__main__":
    main()
