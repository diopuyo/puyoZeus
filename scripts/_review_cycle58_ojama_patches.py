"""cycle 58 で採取された ojama seed patch を画像出力 → 視覚確認.

各動画から ojama label の patch を最大 20 枚 grid 化して出力。
ノイズ混入 (= 文字エフェクト、 背景、 影) を視覚的に確認。
"""
import json
import numpy as np
import cv2
from pathlib import Path

OUT_DIR = Path("data/verify/cycle58_seed_review")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS_TO_REVIEW = ["v89m7", "v35m46", "v43m8", "v50m26", "v78m2", "v31m10"]

def reconstruct_patch(patch_data: dict) -> np.ndarray | None:
    """patch dict → np.ndarray BGR (= label_store 形式想定)."""
    if "data" in patch_data and "shape" in patch_data:
        import base64
        raw = base64.b64decode(patch_data["data"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(patch_data["shape"])
        return arr
    return None

for vid in VIDEOS_TO_REVIEW:
    cf = Path(f"data/phase_l/seeds_cycle58/{vid}/cell.jsonl")
    if not cf.is_file():
        print(f"{vid}: NOT FOUND")
        continue
    ojama_patches = []
    with cf.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("label") != 9:
                continue
            patch = reconstruct_patch(d.get("input_data", {}).get("patch", {}))
            if patch is None:
                continue
            ojama_patches.append(patch)
            if len(ojama_patches) >= 25:
                break
    if not ojama_patches:
        print(f"{vid}: no ojama patches found")
        continue
    # grid 化 (= 5x5)
    n = min(25, len(ojama_patches))
    cols = 5
    rows = (n + cols - 1) // cols
    h, w = ojama_patches[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = i // cols, i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = ojama_patches[i]
    out = OUT_DIR / f"ojama_{vid}.png"
    cv2.imwrite(str(out), grid)
    print(f"{vid}: saved {n} ojama patches → {out}")

print()
print(f"Output dir: {OUT_DIR}")
