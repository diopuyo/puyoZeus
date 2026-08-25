"""W13 (highlight override) の悪化セル実画面フレーム抽出 (診断専用、修正なし)。

`scripts/_score_yardstick_v2_ablation_2026-08-15.py` (w13タグ追加済み) の
c1p vs w13 比較で見つかった「両構成とも match_method=exact (同一フレーム)」の
悪化セル 2 シート (13セル、いずれも単一列がまるごとEMPTYに倒れるパターン)
の実画面を保存する。 出力先: data/verify/diag_w13_yardstick_2026-08-16/

対象:
    000_c109_1P_f652064 (video c109, 1P, col=0 が9セルまるごとEMPTY化)
    002_c11_2P_f54124   (video c11, 2P, col=2 が4セルまるごとEMPTY化)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w13_yardstick_2026-08-16
"""
from __future__ import annotations

import importlib
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
OUT_DIR = _ROOT / "data" / "verify" / "diag_w13_yardstick_2026-08-16"
STD_WIDTH, STD_HEIGHT = 1920, 1080

SHEETS_OF_INTEREST = ["000_c109_1P_f652064", "002_c11_2P_f54124"]

VIDEO_FILENAME_OF: dict[str, str] = {
    "c109": "video_c109.mp4", "c11": "video_c11.mp4",
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
    index_c1p = _sc.load_npz_index(_scab.NPZ_DIRS["c1p"])
    index_w13 = _sc.load_npz_index(_scab.NPZ_DIRS["w13"])

    caps: dict[str, cv2.VideoCapture] = {}

    for sid in SHEETS_OF_INTEREST:
        gt = gt_by_sid[sid]
        vid = gt["video_id"]
        side = gt["side"]
        vfn = VIDEO_FILENAME_OF[vid]
        vpath = VIDEO_DIR / vfn
        cap = caps.setdefault(vid, cv2.VideoCapture(str(vpath)))

        rec_c1p, method_c1p = _sc.match_record(gt, index_c1p)
        rec_w13, method_w13 = _sc.match_record(gt, index_w13)
        if rec_c1p is None or rec_w13 is None:
            print(f"[skip] {sid}: miss (c1p={method_c1p}, w13={method_w13})")
            continue

        reg = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION

        # gt (= 正解ラベル基準フレーム、c1pもexactで同一のはず)
        frame_a = _read_frame(cap, gt["frame_idx"])
        if frame_a is not None:
            roi_a = frame_a[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(str(OUT_DIR / f"{sid}_GT_frame_f{gt['frame_idx']}.png"), frame_a)
            cv2.imwrite(str(OUT_DIR / f"{sid}_GT_roi_f{gt['frame_idx']}.png"), roi_a)

        # w13がマッチしたフレーム (exact同士確認済みなのでgtと同一フレームのはず)
        frame_w13 = _read_frame(cap, rec_w13["frame_idx"])
        if frame_w13 is not None:
            roi_w13 = frame_w13[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(
                str(OUT_DIR / f"{sid}_W13_frame_f{rec_w13['frame_idx']}.png"),
                frame_w13,
            )
            cv2.imwrite(
                str(OUT_DIR / f"{sid}_W13_roi_f{rec_w13['frame_idx']}.png"),
                roi_w13,
            )

        dt = abs(rec_w13["t_sec"] - gt["t_sec"])
        print(f"[done] {sid}: video={vid} side={side} gt_frame={gt['frame_idx']} "
              f"c1p_method={method_c1p} c1p_frame={rec_c1p['frame_idx']} "
              f"w13_method={method_w13} w13_frame={rec_w13['frame_idx']} dt={dt:.3f}s")

    for cap in caps.values():
        cap.release()
    print(f"\n[all done] -> {OUT_DIR}")


if __name__ == "__main__":
    main()
