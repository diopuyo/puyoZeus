"""潜在火力(potential_fire_power)が「build 中の連鎖」を捉えるか検証 (game B)。

reach火力(2手先読み・ツモ依存)は 2Pが build 中で撃てない局面を 0 と読むが、
potential火力(盤面潜在・ツモ非依存)なら 2P の育て中連鎖を捉えられるはず。
game B を流し、各 STABLE 時点で 1P/2P の reach と potential を並べ、
特に後半(2Pが勝っていた区間)で potential が 2P 優位を示すかを見る。
"""
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
START, END, WARM = 202.0, 283.0, 16.0


def _fp(side, board, elapsed):
    """(reach お邪魔, potential お邪魔) を返す。"""
    reach = iv.reach_fire_power(board, side.next_pair, side.dnext_pair, elapsed).value.raw
    pot = iv.potential_fire_power(board, elapsed).raw
    return reach, pot


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
    print(" t_rel | reach 1P/2P | potential 1P/2P | reach差 pot差")
    rows = []
    for fi in range(int((START - WARM) * fps), int(END * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080))
        if fi % step:
            continue
        t = fi / fps
        r = pipe.update(fi, t, f)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1s = r.p1.confirmed_board
        else:
            b1s = None
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2s = r.p2.confirmed_board
        else:
            b2s = None
        b1 = b1s or b1; b2 = b2s or b2
        _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t, tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if t < START or b1 is None or b2 is None:
            continue
        el = tr._elapsed(t)
        rc1, pt1 = _fp(r.p1, b1, el); rc2, pt2 = _fp(r.p2, b2, el)
        rows.append((t - START, rc1, rc2, pt1, pt2))
        print(f"{t - START:6.1f} | {rc1:4.0f}/{rc2:4.0f}   | {pt1:5.0f}/{pt2:5.0f}     "
              f"| {rc1 - rc2:+5.0f} {pt1 - pt2:+5.0f}")
    cap.release()
    late = [r for r in rows if r[0] >= 30]
    if late:
        rd = sum(r[1] - r[2] for r in late) / len(late)
        pd = sum(r[3] - r[4] for r in late) / len(late)
        print(f"\n後半(t_rel>=30s) 平均: reach差={rd:+.1f}  potential差={pd:+.1f}"
              f"  (2P勝ちなら負が正しい)")


if __name__ == "__main__":
    main()
