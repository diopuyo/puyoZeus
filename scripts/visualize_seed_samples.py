"""seed dataset の各色からランダムサンプル patch を画像化 (ユーザー目視用).

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_seed_samples \
        --seed-root data/pseudo_labels_hsv_seed/v29 \
        --output data/seed_review/v29_samples.png \
        --per-color 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.self_supervised.pseudo_label import PseudoLabelSample

COLOR_NAMES: dict[int, str] = {
    1: "red", 2: "blue", 3: "green",
    4: "yellow", 5: "purple", 9: "ojama",
}


def load_samples(seed_root: Path) -> list[PseudoLabelSample]:
    jsonl_path = seed_root / "cell.jsonl"
    if not jsonl_path.exists():
        return []
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(PseudoLabelSample.from_jsonable(obj))
            except Exception:
                continue
    return samples


def build_grid(
    samples: list[PseudoLabelSample], per_color: int, patch_size: int = 64,
) -> np.ndarray:
    by_color: dict[int, list[np.ndarray]] = {}
    for s in samples:
        color = int(s.label)
        if color not in COLOR_NAMES:
            continue
        patch = s.input_data
        if isinstance(patch, dict):
            patch = patch.get("patch")
        if not isinstance(patch, np.ndarray):
            continue
        by_color.setdefault(color, []).append(patch)
    # cycle 32 (2026-05-19): 採取された色のみ表示。 ojama を skip した場合等で
    # 空の dummy row + ラベル「(n=0)」 が出力される問題への対策。
    rows = []
    for color in (1, 2, 3, 4, 5, 9):
        patches = by_color.get(color, [])
        if not patches:
            continue
        sampled = random.sample(patches, min(per_color, len(patches)))
        resized = []
        for p in sampled:
            if p.size == 0:
                resized.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
            else:
                resized.append(cv2.resize(p, (patch_size, patch_size)))
        while len(resized) < per_color:
            resized.append(np.full((patch_size, patch_size, 3), 32, dtype=np.uint8))
        row = np.hstack(resized)
        label_strip = np.full((24, row.shape[1], 3), 0, dtype=np.uint8)
        cv2.putText(
            label_strip,
            f"{COLOR_NAMES[color]} (n={len(patches)})",
            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
        rows.append(np.vstack([label_strip, row]))
    if not rows:
        raise RuntimeError("no patches available in any trainable color")
    return np.vstack(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-color", type=int, default=20)
    p.add_argument("--patch-size", type=int, default=64)
    args = p.parse_args()
    samples = load_samples(args.seed_root)
    if not samples:
        print(f"[error] no samples in {args.seed_root}")
        return 1
    print(f"[info] loaded {len(samples)} samples from {args.seed_root}")
    counts: dict[int, int] = {}
    for s in samples:
        color = int(s.label)
        counts[color] = counts.get(color, 0) + 1
    for c, n in sorted(counts.items()):
        name = COLOR_NAMES.get(c, str(c))
        print(f"  color={c} ({name}): {n} samples")
    grid = build_grid(samples, args.per_color, args.patch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid)
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
