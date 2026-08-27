"""物理制約vs画素差分ゲート比較の証拠フレーム抽出 (2026-08-18)。

logs/_diag_physics_vs_pixel_gate_2026-08-18_merged.json から
「Bのみ (画素不安定だが物理は整合=不当に捨てられていた候補)」の代表例を
各カテゴリから数件サンプリングし、該当 frame_idx 近傍の実画面 PNG を保存する。

実装しない (収集フィルタは変更しない)。診断用の証拠保存のみ。
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MERGED_JSON = Path("logs/_diag_physics_vs_pixel_gate_2026-08-18_merged.json")
FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("logs/_diag_pvpg_evidence_2026-08-18")
TARGET_W, TARGET_H = 1920, 1080
N_PER_CATEGORY = 2

random.seed(20260818)


def main() -> None:
    data = json.loads(MERGED_JSON.read_text(encoding="utf-8"))
    rows = data["rows"]
    b_only = [
        r for r in rows
        if r["b_stable"] is not None and (not r["a_violation"]) and (not r["b_stable"])
    ]
    by_cat: "dict[str, list[dict]]" = defaultdict(list)
    for r in b_only:
        by_cat[r["category"]].append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caps: "dict[str, cv2.VideoCapture]" = {}
    for cat, cat_rows in by_cat.items():
        sample = random.sample(cat_rows, min(N_PER_CATEGORY, len(cat_rows)))
        for r in sample:
            vid = r["video"]
            if vid not in caps:
                caps[vid] = cv2.VideoCapture(str(FRAMES_DIR / f"video_{vid}.mp4"))
            cap = caps[vid]
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, r["frame_idx"]))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[skip] {vid} frame={r['frame_idx']} 読み込み失敗")
                continue
            if frame.shape[:2] != (TARGET_H, TARGET_W):
                frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            name = (
                f"{vid}_{r['side']}_g{r['game_idx']}_f{r['frame_idx']}_"
                f"t{r['t_sec']:.1f}_{cat}_diffmax{r['b_diff_max_in_window']:.2f}.png"
            )
            out_path = OUT_DIR / name
            cv2.imwrite(str(out_path), frame)
            print(f"[保存] {out_path}")
    for cap in caps.values():
        cap.release()


if __name__ == "__main__":
    main()
