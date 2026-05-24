"""HSV-seed dataset の色別 patch を grid 画像化 (目視レビュー用)。

各色 32 件 (8x4 grid) を random sample、 1 PNG/色で出力。
patches は元サイズ → 4x 拡大表示。

出力:
    data/pseudo_labels_hsv_seed/review_{color_label}_{color_name}.png
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

from src.self_supervised.pseudo_label import _decode_ndarray

COLOR_NAMES: dict[int, str] = {
    1: "red", 2: "blue", 3: "green",
    4: "yellow", 5: "purple", 9: "ojama",
}

GRID_COLS: int = 8
GRID_ROWS: int = 4
PATCHES_PER_COLOR: int = GRID_COLS * GRID_ROWS  # 32
PATCH_DISPLAY_SIZE: int = 80  # 拡大表示サイズ (元 ~20x20 → 80x80)


def _load_patches_by_color(
    seed_root: Path, video_ids: list[str],
) -> dict[int, list[tuple[np.ndarray, str]]]:
    """色別に patch を集計 (= (patch, video_id) のリスト)。"""
    by_color: dict[int, list[tuple[np.ndarray, str]]] = {}
    for vid in video_ids:
        jsonl = seed_root / vid / "cell.jsonl"
        if not jsonl.exists():
            continue
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                label = int(obj["label"])
                inp = obj.get("input_data", {})
                if not isinstance(inp, dict):
                    continue
                patch_data = inp.get("patch")
                if not isinstance(patch_data, dict):
                    continue
                if patch_data.get("__ndarray__") is not True:
                    continue
                patch = _decode_ndarray(patch_data)
                by_color.setdefault(label, []).append((patch, vid))
    return by_color


def _build_grid(
    patches: list[tuple[np.ndarray, str]], color_label: int,
) -> np.ndarray:
    """1 色分の patch grid 画像を構築 (8x4 = 32 件、 余白に video_id 表記)。"""
    cell_w = PATCH_DISPLAY_SIZE
    cell_h = PATCH_DISPLAY_SIZE + 14  # 下に video_id ラベル用
    grid = np.full(
        (GRID_ROWS * cell_h, GRID_COLS * cell_w, 3),
        40, dtype=np.uint8,
    )
    for i, (patch, vid) in enumerate(patches[:PATCHES_PER_COLOR]):
        r = i // GRID_COLS
        c = i % GRID_COLS
        resized = cv2.resize(
            patch, (cell_w, PATCH_DISPLAY_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        y0 = r * cell_h
        x0 = c * cell_w
        grid[y0:y0 + PATCH_DISPLAY_SIZE, x0:x0 + cell_w] = resized
        # video_id label
        cv2.putText(
            grid, vid, (x0 + 2, y0 + PATCH_DISPLAY_SIZE + 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
    # タイトル帯 (上に 24 px)
    title_h = 24
    title = np.full((title_h, grid.shape[1], 3), 80, dtype=np.uint8)
    cv2.putText(
        title,
        f"color={color_label} ({COLOR_NAMES.get(color_label, '?')})  "
        f"shown={min(len(patches), PATCHES_PER_COLOR)}/total={len(patches)}",
        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        cv2.LINE_AA,
    )
    return np.vstack([title, grid])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seed-root", type=Path,
        default=Path("data/pseudo_labels_hsv_seed"),
    )
    p.add_argument(
        "--video-ids", nargs="+",
        default=["v97", "v70", "v89m3", "v50", "v91"],
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    by_color = _load_patches_by_color(args.seed_root, args.video_ids)
    if not by_color:
        print("[review] no patches found", file=sys.stderr)
        return 1
    out_paths: list[Path] = []
    for color_label in sorted(by_color.keys()):
        patches = by_color[color_label]
        random.shuffle(patches)
        grid_img = _build_grid(patches, color_label)
        name = COLOR_NAMES.get(color_label, f"c{color_label}")
        out = args.seed_root / f"review_{color_label}_{name}.png"
        cv2.imwrite(str(out), grid_img)
        out_paths.append(out)
        print(f"[review] {out} (total {len(patches)})")
    print("---all---")
    for p in out_paths:
        print(p.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
