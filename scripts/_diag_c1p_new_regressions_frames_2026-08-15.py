"""c1p (placement_override 修正版) の新規劣化25セルの実画面フレーム抽出

(診断専用、修正なし、`scripts/_diag_placement_override_frames_2026-08-15.py`
の c1p 版)。

各劣化盤面について、
  (a) baseline(a)がSTABLE突合に使ったフレーム (= gt frame_idx、exact match)
  (b) c1pがSTABLE突合に使ったフレーム (exact一致ならaと同一、nearestならズレたfr)
の両方を生フレーム全体+盤面ROIで保存する。 出力先は data/verify/
yardstick_v2_2026-08-14/diag_c1p/ (既存 diag_c1/ とは独立)。

対象シート (25セル、`_score_yardstick_v2_ablation_2026-08-15.py --compare
--compare-tags a c1p` のペアワイズ new_regression 集計で確定):
  010_c96_2P_f951622 (12セル), 026_c13_2P_f17462 (3), 013_c13_2P_f17458 (2),
  019_c23_1P_f150153 (3), 006_c17_2P_f17006 (5)

Usage:
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._diag_c1p_new_regressions_frames_2026-08-15
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
OUT_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "diag_c1p"
STD_WIDTH, STD_HEIGHT = 1920, 1080

SHEETS_OF_INTEREST = [
    "010_c96_2P_f951622", "026_c13_2P_f17462", "013_c13_2P_f17458",
    "019_c23_1P_f150153", "006_c17_2P_f17006",
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


def compute_new_regressions() -> list[dict]:
    """a vs c1p のペアワイズ new_regression セル一覧を計算する

    (`_score_yardstick_v2_ablation_2026-08-15.py` の (3) 誤り分類ロジックと
    同一の考え方、 セル単位の明細が必要なため個別に再計算する)。
    """
    a_rows = {
        r["sheet_id"]: r for r in json.loads(
            (_scab.SCORING_DIR / "score_a.json").read_text(encoding="utf-8"),
        )
    }
    c1p_rows = {
        r["sheet_id"]: r for r in json.loads(
            (_scab.SCORING_DIR / "score_c1p.json").read_text(encoding="utf-8"),
        )
    }
    pair_common = set(a_rows) & set(c1p_rows)
    out: list[dict] = []
    for sid in pair_common:
        a_cells = {(c["r"], c["c"]): c for c in a_rows[sid].get("cells", [])}
        c1p_cells = {(c["r"], c["c"]): c for c in c1p_rows[sid].get("cells", [])}
        for key, ac in a_cells.items():
            cc = c1p_cells.get(key)
            if cc is None:
                continue
            wrong_a = not ac["is_correct"]
            wrong_c1p = not cc["is_correct"]
            if (not wrong_a) and wrong_c1p:
                out.append({
                    "sheet_id": sid, "r": key[0], "c": key[1],
                    "correct": ac["correct"], "a_pred": ac["pred"],
                    "c1p_pred": cc["pred"],
                    "video_id": a_rows[sid]["video_id"],
                    "side": a_rows[sid]["side"], "phase": a_rows[sid]["phase"],
                })
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gts = _sc.load_ground_truth()
    gt_by_sid = {g["sheet_id"]: g for g in gts}
    index_c1p = _sc.load_npz_index(_scab.NPZ_DIRS["c1p"])

    regressions = compute_new_regressions()
    reg_out_path = (
        _scab.SCORING_DIR / "_diag_c1p_new_regressions_2026-08-15.json"
    )
    reg_out_path.write_text(
        json.dumps(regressions, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[info] new_regression cells (a vs c1p pairwise) = {len(regressions)} "
          f"-> {reg_out_path}")

    caps: dict[str, cv2.VideoCapture] = {}

    for sid in SHEETS_OF_INTEREST:
        gt = gt_by_sid[sid]
        vid = gt["video_id"]
        side = gt["side"]
        vfn = VIDEO_FILENAME_OF[vid]
        vpath = VIDEO_DIR / vfn
        cap = caps.setdefault(vid, cv2.VideoCapture(str(vpath)))

        rec_c1p, method_c1p = _sc.match_record(gt, index_c1p)
        if rec_c1p is None:
            print(f"[skip] {sid}: c1p側miss")
            continue

        reg = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION

        # (a) gt (=baseline a の exact match) フレーム
        frame_a = _read_frame(cap, gt["frame_idx"])
        if frame_a is not None:
            roi_a = frame_a[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(str(OUT_DIR / f"{sid}_A_frame_f{gt['frame_idx']}.png"), frame_a)
            cv2.imwrite(str(OUT_DIR / f"{sid}_A_roi_f{gt['frame_idx']}.png"), roi_a)

        # (b) c1pがマッチしたフレーム (nearestならgtとフレーム番号が異なる)
        frame_c1p = _read_frame(cap, rec_c1p["frame_idx"])
        if frame_c1p is not None:
            roi_c1p = frame_c1p[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(
                str(OUT_DIR / f"{sid}_C1P_frame_f{rec_c1p['frame_idx']}.png"),
                frame_c1p,
            )
            cv2.imwrite(
                str(OUT_DIR / f"{sid}_C1P_roi_f{rec_c1p['frame_idx']}.png"),
                roi_c1p,
            )

        dt = abs(rec_c1p["t_sec"] - gt["t_sec"])
        print(f"[done] {sid}: video={vid} side={side} gt_frame={gt['frame_idx']} "
              f"c1p_method={method_c1p} c1p_frame={rec_c1p['frame_idx']} dt={dt:.3f}s")

    for cap in caps.values():
        cap.release()
    print(f"\n[all done] -> {OUT_DIR}")


if __name__ == "__main__":
    main()
