"""W10 (赤→紫) 根因: HSV再現では真値通り紫と出るのに本番pred=赤 だった18セルについて、
本番採用CNN (models/cnn_phase_b_large_v2.pt, cnn_override_prob=0.70) が実際に
どう推論するかを検証する (2026-08-17)。本体コード変更なし・計装のみ。

出力: data/verify/diag_w10_2026-08-17/cnn_verify.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.patch_classifier import CLASS_INDEX_TO_COLOR, CnnPatchClassifierLarge

_ROOT = Path(__file__).resolve().parent.parent
YARDSTICK_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14"
ANCHORS_DIR = YARDSTICK_DIR / "anchors"
SCORE_C1P = YARDSTICK_DIR / "scoring_ablation" / "score_c1p.json"
OUT_DIR = _ROOT / "data" / "verify" / "diag_w10_2026-08-17"
MODEL_PATH = _ROOT / "models" / "cnn_phase_b_large_v2.pt"

COLOR_NAME = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "おじゃま", 10: "不明"}


def _region_for_side(side: str):
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def main() -> None:
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cnn = CnnPatchClassifierLarge()
    state = torch.load(MODEL_PATH, map_location="cpu")
    cnn._model.load_state_dict(state)
    cnn._model.eval()

    scores = json.loads(SCORE_C1P.read_text(encoding="utf-8"))
    rows_out = []
    for sheet in scores:
        if "cells" not in sheet:
            continue
        sheet_id = sheet["sheet_id"]
        side = sheet["side"]
        frame_path = ANCHORS_DIR / f"{sheet_id}_frame.png"
        if not frame_path.exists():
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        for cell in sheet["cells"]:
            correct = cell["correct"]
            pred_prod = cell["pred"]
            if correct not in (1, 5):
                continue
            is_w10 = (correct == 5 and pred_prod == 1)
            is_ref_purple_ok = (correct == 5 and pred_prod == 5)
            is_ref_red_ok = (correct == 1 and pred_prod == 1)
            if not (is_w10 or is_ref_purple_ok or is_ref_red_ok):
                continue
            region = _region_for_side(side)
            x1, y1, x2, y2 = region.cell_sample_rect(cell["r"], cell["c"])
            patch = frame[y1:y2, x1:x2]
            probs = cnn.predict_proba(patch)
            best_idx = int(np.argmax(probs))
            cnn_color = CLASS_INDEX_TO_COLOR[best_idx]
            cnn_prob = float(probs[best_idx])
            # 赤/紫の個別確率も記録 (色コード->クラスindex 逆引き)
            color_to_idx = {c: i for i, c in enumerate(CLASS_INDEX_TO_COLOR)}
            p_red = float(probs[color_to_idx[1]]) if 1 in color_to_idx else float("nan")
            p_purple = float(probs[color_to_idx[5]]) if 5 in color_to_idx else float("nan")
            category = (
                "W10_error(真紫→prod pred赤)" if is_w10 else
                "ref_purple_correct" if is_ref_purple_ok else
                "ref_red_correct"
            )
            rows_out.append({
                "sheet_id": sheet_id, "side": side, "r": cell["r"], "c": cell["c"],
                "correct": correct, "pred_prod": pred_prod, "category": category,
                "cnn_argmax_color": cnn_color, "cnn_argmax_prob": round(cnn_prob, 4),
                "cnn_p_red": round(p_red, 4), "cnn_p_purple": round(p_purple, 4),
                "cnn_overrides(>=0.70)": cnn_prob >= 0.70,
            })

    out_csv = OUT_DIR / "cnn_verify.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[done] {len(rows_out)} 件 -> {out_csv}")

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows_out:
        by_cat[r["category"]].append(r)
    for cat, rs in by_cat.items():
        print(f"\n=== {cat} (n={len(rs)}) ===")
        for r in rs:
            print(f"  {r['sheet_id']} r{r['r']}c{r['c']}: cnn_argmax={COLOR_NAME.get(r['cnn_argmax_color'])}"
                  f"({r['cnn_argmax_prob']}) p_red={r['cnn_p_red']} p_purple={r['cnn_p_purple']} "
                  f"override={r['cnn_overrides(>=0.70)']}")


if __name__ == "__main__":
    main()
