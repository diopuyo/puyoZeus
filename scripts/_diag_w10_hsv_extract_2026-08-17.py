"""W10 (赤→紫 系統的色誤り) 根因診断: 誤りセルの実HSV値を計装抽出する (2026-08-17)。

本体コード変更なし。 物差しv2の参照フレームPNG (anchors/*_frame.png) から
本番と同一の ColorClassifier ロジック (median H (赤2峰折返し補正込み) /
median S (光沢除外OFF=デフォルト) / median V) を再現して、
W10誤りセル (真=紫・現行本番pred=赤) の HSV 実測値を出力する。
比較用に「正しく紫と分類されたセル」「正しく赤と分類されたセル」のHSVも
同時に集計し、閾値境界との位置関係を明らかにする。

出力: data/verify/diag_w10_2026-08-17/hsv_measurements.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ColorClassifier,
    _median_fast,
)

_ROOT = Path(__file__).resolve().parent.parent
YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
ANCHORS_DIR = YARDSTICK_DIR / "anchors"
SCORE_C1P = YARDSTICK_DIR / "scoring_ablation" / "score_c1p.json"
OUT_DIR = _ROOT / "data" / "verify" / "diag_w10_2026-08-17"

RED_HUE_WRAP_THRESHOLD = 140  # image_reader.py に定義される定数と合わせる（下でimportに失敗したら手動値）


def _region_for_side(side: str):
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def _measure_cell(frame_bgr: np.ndarray, side: str, r: int, c: int) -> dict:
    """本番 classify() と同一の median H/S/V (wrap補正込み・specular補正OFF) を再現計測する。"""
    region = _region_for_side(side)
    x1, y1, x2, y2 = region.cell_sample_rect(r, c)
    patch = frame_bgr[y1:y2, x1:x2]
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    clf = ColorClassifier()  # default: vote_mode=False, enable_red_hue_wrap_fix=True (本番既定と一致)
    h = clf._compute_stable_h_median(hsv_patch[:, :, 0])
    s = clf._compute_specular_robust_s(hsv_patch[:, :, 1], hsv_patch[:, :, 2])
    v = int(_median_fast(np.asarray(hsv_patch[:, :, 2]).ravel()))
    h_raw = int(_median_fast(np.asarray(hsv_patch[:, :, 0]).ravel()))
    pred = clf.classify(patch)
    return {
        "h_median": h, "h_raw_median": h_raw, "s_median": s, "v_median": v,
        "reclassify": pred, "patch_shape": patch.shape[:2],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = json.loads(SCORE_C1P.read_text(encoding="utf-8"))

    rows_out = []
    for sheet in scores:
        if "cells" not in sheet:
            continue
        sheet_id = sheet["sheet_id"]
        side = sheet["side"]
        frame_path = ANCHORS_DIR / f"{sheet_id}_frame.png"
        if not frame_path.exists():
            print(f"[warn] frame画像なし: {frame_path}")
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[warn] 読み込み失敗: {frame_path}")
            continue
        for cell in sheet["cells"]:
            correct = cell["correct"]
            pred_prod = cell["pred"]
            # 対象: (1) W10当該セル(真紫・現行pred赤)、(2) 比較対照=正しく紫と判定されたセル、
            # (3) 比較対照=正しく赤と判定されたセル (紫/赤どちらかが正解のセルのみ抽出)
            if correct not in (1, 5):
                continue
            category = None
            if correct == 5 and pred_prod == 1:
                category = "W10_error(真紫→pred赤)"
            elif correct == 5 and pred_prod == 5:
                category = "ref_purple_correct"
            elif correct == 1 and pred_prod == 1:
                category = "ref_red_correct"
            elif correct == 1 and pred_prod == 5:
                category = "reverse_error(真赤→pred紫、あれば要注目)"
            else:
                category = f"other(correct={correct},pred={pred_prod})"
            m = _measure_cell(frame, side, cell["r"], cell["c"])
            rows_out.append({
                "sheet_id": sheet_id, "video_id": sheet["video_id"], "side": side,
                "r": cell["r"], "c": cell["c"], "correct": correct, "pred_prod": pred_prod,
                "category": category,
                **{k: v for k, v in m.items() if k != "patch_shape"},
                "patch_h": m["patch_shape"][0], "patch_w": m["patch_shape"][1],
            })

    out_csv = OUT_DIR / "hsv_measurements.csv"
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[done] {len(rows_out)} 件 -> {out_csv}")

    # サマリ表示
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows_out:
        by_cat[r["category"]].append(r)
    for cat, rs in by_cat.items():
        print(f"\n=== {cat} (n={len(rs)}) ===")
        for r in rs:
            print(f"  {r['sheet_id']} r{r['r']}c{r['c']}: "
                  f"H={r['h_median']}(raw={r['h_raw_median']}) S={r['s_median']} V={r['v_median']} "
                  f"reclassify={r['reclassify']}")


if __name__ == "__main__":
    main()
