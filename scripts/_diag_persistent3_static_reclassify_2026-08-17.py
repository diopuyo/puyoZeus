"""持続誤認3件 (c23/c10/c109) の静的再分類診断 (2026-08-17)。本体コード変更なし・計装のみ。

anchor frame (物差しv2参照フレームPNG) から対象セルのパッチを切り出し、
本番同一ロジックの HSV-only 分類器 (ColorClassifier) と CNN (CnnPatchClassifierLarge,
models/cnn_phase_b_large_v2.pt) それぞれで単独再分類する。
「分類器自体が誤る (=classifier bug)」か「分類器は正しいのに確定盤面が誤り
(=上流の状態機械/投票ロジックの問題)」かを切り分ける。

出力: data/verify/diag_persistent3_2026-08-17/static_reclassify.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, ColorClassifier, _median_fast
from src.patch_classifier import CLASS_INDEX_TO_COLOR, CnnPatchClassifierLarge

_ROOT = Path(__file__).resolve().parent.parent
ANCHORS_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "anchors"
OUT_DIR = _ROOT / "data" / "verify" / "diag_persistent3_2026-08-17"
MODEL_PATH = _ROOT / "models" / "cnn_phase_b_large_v2.pt"

COLOR_NAME = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "おじゃま", 10: "不明"}

TARGETS = {
    "007_c23_2P_f84251": {
        "side": "2P", "wrong": 1, "correct": 5,
        "cells": [(7, 1), (8, 4), (9, 5), (10, 0), (10, 1), (10, 2), (10, 5), (11, 4), (12, 3), (12, 4)],
    },
    "009_c10_2P_f80448": {
        "side": "2P", "wrong": 5, "correct": 2,
        "cells": [(2, 5), (3, 5), (4, 1), (4, 4), (5, 1), (5, 4), (6, 0), (7, 0), (7, 3),
                  (8, 3), (8, 4), (10, 2), (11, 3), (11, 4), (12, 2)],
    },
    "000_c109_1P_f652064": {
        "side": "1P", "wrong": 9, "correct": 1,
        "cells": [(4, 4)],
    },
}


def _region_for_side(side: str):
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cnn = CnnPatchClassifierLarge()
    state = torch.load(MODEL_PATH, map_location="cpu")
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    hsv_clf = ColorClassifier()
    color_to_idx = {c: i for i, c in enumerate(CLASS_INDEX_TO_COLOR)}

    rows_out = []
    for sheet_id, cfg in TARGETS.items():
        frame_path = ANCHORS_DIR / f"{sheet_id}_frame.png"
        frame = cv2.imread(str(frame_path))
        assert frame is not None, f"読み込み失敗: {frame_path}"
        region = _region_for_side(cfg["side"])
        for (r, c) in cfg["cells"]:
            x1, y1, x2, y2 = region.cell_sample_rect(r, c)
            patch = frame[y1:y2, x1:x2]
            hsv_pred = hsv_clf.classify(patch)
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h_med = int(_median_fast(np.asarray(hsv_patch[:, :, 0]).ravel()))
            s_med = int(_median_fast(np.asarray(hsv_patch[:, :, 1]).ravel()))
            v_med = int(_median_fast(np.asarray(hsv_patch[:, :, 2]).ravel()))
            probs = cnn.predict_proba(patch)
            best_idx = int(np.argmax(probs))
            cnn_pred = CLASS_INDEX_TO_COLOR[best_idx]
            cnn_prob = float(probs[best_idx])
            p_wrong = float(probs[color_to_idx[cfg["wrong"]]]) if cfg["wrong"] in color_to_idx else float("nan")
            p_correct = float(probs[color_to_idx[cfg["correct"]]]) if cfg["correct"] in color_to_idx else float("nan")
            rows_out.append({
                "sheet_id": sheet_id, "r": r, "c": c,
                "wrong_value": cfg["wrong"], "correct_value": cfg["correct"],
                "hsv_pred": hsv_pred, "hsv_pred_name": COLOR_NAME.get(hsv_pred),
                "h_med": h_med, "s_med": s_med, "v_med": v_med,
                "cnn_pred": cnn_pred, "cnn_pred_name": COLOR_NAME.get(cnn_pred),
                "cnn_prob": round(cnn_prob, 4),
                "cnn_p_wrong": round(p_wrong, 4), "cnn_p_correct": round(p_correct, 4),
                "hsv_matches_wrong": hsv_pred == cfg["wrong"],
                "hsv_matches_correct": hsv_pred == cfg["correct"],
                "cnn_matches_wrong": cnn_pred == cfg["wrong"],
                "cnn_matches_correct": cnn_pred == cfg["correct"],
            })

    out_csv = OUT_DIR / "static_reclassify.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[done] {len(rows_out)} 件 -> {out_csv}")

    from collections import defaultdict
    by_sheet = defaultdict(list)
    for r in rows_out:
        by_sheet[r["sheet_id"]].append(r)
    for sheet_id, rs in by_sheet.items():
        n_hsv_wrong = sum(1 for r in rs if r["hsv_matches_wrong"])
        n_hsv_correct = sum(1 for r in rs if r["hsv_matches_correct"])
        n_cnn_wrong = sum(1 for r in rs if r["cnn_matches_wrong"])
        n_cnn_correct = sum(1 for r in rs if r["cnn_matches_correct"])
        print(f"\n=== {sheet_id} (n={len(rs)}) ===")
        print(f"  HSVが誤り値と一致={n_hsv_wrong}/{len(rs)}  HSVが正解値と一致={n_hsv_correct}/{len(rs)}")
        print(f"  CNNが誤り値と一致={n_cnn_wrong}/{len(rs)}  CNNが正解値と一致={n_cnn_correct}/{len(rs)}")
        for r in rs:
            print(f"    r{r['r']}c{r['c']}: HSV={r['hsv_pred_name']}(H={r['h_med']},S={r['s_med']},V={r['v_med']}) "
                  f"CNN={r['cnn_pred_name']}({r['cnn_prob']}) p_wrong={r['cnn_p_wrong']} p_correct={r['cnn_p_correct']}")


if __name__ == "__main__":
    main()
