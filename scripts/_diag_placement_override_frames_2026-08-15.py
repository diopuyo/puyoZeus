"""placement_override(c1)新規劣化20セルの実画面フレーム抽出 (診断専用、修正なし)。

各劣化盤面について、
  (a) baseline(a)がSTABLE突合に使ったフレーム (= gt frame_idx、exact match)
  (b) c1がSTABLE突合に使ったフレーム (exact一致ならaと同一、nearestならズレたfr)
の両方を生フレーム全体+盤面ROIで保存し、
c1のSTABLE時刻が実際にずれているか (タイミングずれ仮説) を実画面で確認できるようにする。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

_sc = importlib.import_module("scripts._score_yardstick_v2_2026-08-14")
_scab = importlib.import_module("scripts._score_yardstick_v2_ablation_2026-08-15")

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "diag_c1"
STD_WIDTH, STD_HEIGHT = 1920, 1080

SHEETS_OF_INTEREST = [
    "003_c17_2P_f17284", "010_c96_2P_f951622", "013_c13_2P_f17458",
    "019_c23_1P_f150153", "026_c13_2P_f17462", "035_c10_1P_f80816",
]

VIDEO_FILENAME_OF: dict[str, str] = {
    "c17": "video_c17.mp4", "c96": "_hold_video_c96.mp4", "c13": "video_c13.mp4",
    "c23": "video_c23.mp4", "c10": "video_c10.mp4",
}


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> "np.ndarray | None":
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (STD_HEIGHT, STD_WIDTH):
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gts = _sc.load_ground_truth()
    gt_by_sid = {g["sheet_id"]: g for g in gts}
    index_c1 = _sc.load_npz_index(_scab.NPZ_DIRS["c1"])

    caps: dict[str, cv2.VideoCapture] = {}

    for sid in SHEETS_OF_INTEREST:
        gt = gt_by_sid[sid]
        vid = gt["video_id"]
        side = gt["side"]
        vfn = VIDEO_FILENAME_OF[vid]
        vpath = VIDEO_DIR / vfn
        cap = caps.setdefault(vid, cv2.VideoCapture(str(vpath)))

        rec_c1, method_c1 = _sc.match_record(gt, index_c1)
        if rec_c1 is None:
            print(f"[skip] {sid}: c1側miss")
            continue

        reg = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION

        # (a) gt (=baseline a の exact match) フレーム
        frame_a = _read_frame(cap, gt["frame_idx"])
        if frame_a is not None:
            roi_a = frame_a[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(str(OUT_DIR / f"{sid}_A_frame_f{gt['frame_idx']}.png"), frame_a)
            cv2.imwrite(str(OUT_DIR / f"{sid}_A_roi_f{gt['frame_idx']}.png"), roi_a)

        # (b) c1がマッチしたフレーム (nearestならgtとフレーム番号が異なる)
        frame_c1 = _read_frame(cap, rec_c1["frame_idx"])
        if frame_c1 is not None:
            roi_c1 = frame_c1[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(str(OUT_DIR / f"{sid}_C1_frame_f{rec_c1['frame_idx']}.png"), frame_c1)
            cv2.imwrite(str(OUT_DIR / f"{sid}_C1_roi_f{rec_c1['frame_idx']}.png"), roi_c1)

        dt = abs(rec_c1["t_sec"] - gt["t_sec"])
        print(f"[done] {sid}: video={vid} side={side} gt_frame={gt['frame_idx']} "
              f"c1_method={method_c1} c1_frame={rec_c1['frame_idx']} dt={dt:.3f}s")

    for cap in caps.values():
        cap.release()
    print(f"\n[all done] -> {OUT_DIR}")


if __name__ == "__main__":
    main()
