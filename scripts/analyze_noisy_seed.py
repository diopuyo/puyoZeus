"""NG seed の metadata 分布 + 単色 patch 大量表示 (cycle 32 F + G).

F: 各色の patch の metadata (row, col, side, frame_idx) を集計
G: 単色 patch を grid 表示 (= per-color 100 枚等)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.analyze_noisy_seed \
        --seed-root data/pseudo_labels_hsv_seed/v29_old_noisy \
        --out-root data/seed_review
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    jsonl = seed_root / "cell.jsonl"
    if not jsonl.exists():
        return []
    out = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out.append(PseudoLabelSample.from_jsonable(obj))
            except Exception:
                continue
    return out


def analyze_metadata(
    samples: list[PseudoLabelSample], focus_colors: list[int],
) -> str:
    """各色について row/col/side/frame_idx の分布を集計."""
    report = []
    for color in focus_colors:
        name = COLOR_NAMES.get(color, str(color))
        targets = [s for s in samples if int(s.label) == color]
        if not targets:
            report.append(f"=== {name} (color={color}): 0 samples ===")
            continue
        report.append(
            f"=== {name} (color={color}): {len(targets)} samples ===",
        )
        rows = Counter(int(s.metadata.get("row", -1)) for s in targets)
        cols = Counter(int(s.metadata.get("col", -1)) for s in targets)
        sides = Counter(str(s.metadata.get("side", "?")) for s in targets)
        frames = [int(s.metadata.get("frame_idx", -1)) for s in targets]
        report.append(
            f"  row distribution: {sorted(rows.items())}",
        )
        report.append(
            f"  col distribution: {sorted(cols.items())}",
        )
        report.append(
            f"  side distribution: {sorted(sides.items())}",
        )
        if frames:
            f_min, f_max = min(frames), max(frames)
            f_med = sorted(frames)[len(frames) // 2]
            report.append(
                f"  frame_idx: min={f_min} median={f_med} max={f_max} "
                f"(range={f_max - f_min})",
            )
        report.append("")
    return "\n".join(report)


def build_single_color_grid(
    samples: list[PseudoLabelSample], color: int,
    per_grid: int, patch_size: int = 48, cols_per_row: int = 20,
) -> np.ndarray | None:
    """単色 patch を grid 表示 (per_grid 枚)."""
    targets = []
    for s in samples:
        if int(s.label) != color:
            continue
        patch = s.input_data
        if isinstance(patch, dict):
            patch = patch.get("patch")
        if isinstance(patch, np.ndarray):
            targets.append(patch)
        if len(targets) >= per_grid:
            break
    if not targets:
        return None
    rows_count = (len(targets) + cols_per_row - 1) // cols_per_row
    grid_h = rows_count * patch_size + 30
    grid_w = cols_per_row * patch_size
    img = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)
    cv2.putText(
        img, f"{COLOR_NAMES[color]} patches (n={len(targets)})",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
    )
    for i, patch in enumerate(targets):
        r = i // cols_per_row
        c = i % cols_per_row
        resized = cv2.resize(patch, (patch_size, patch_size))
        y0 = 30 + r * patch_size
        x0 = c * patch_size
        img[y0:y0 + patch_size, x0:x0 + patch_size] = resized
    return img


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, default=Path("data/seed_review"))
    p.add_argument(
        "--focus-colors", type=int, nargs="+", default=[4, 9, 1, 2],
        help="metadata 分析 + grid 表示する色 (yellow=4, ojama=9, red=1, blue=2)",
    )
    p.add_argument("--per-grid", type=int, default=100)
    args = p.parse_args()

    samples = load_samples(args.seed_root)
    if not samples:
        print(f"[error] no samples in {args.seed_root}")
        return 1
    print(f"[info] loaded {len(samples)} samples")

    args.out_root.mkdir(parents=True, exist_ok=True)
    # F: metadata 分布解析 (= テキスト report)
    report = analyze_metadata(samples, args.focus_colors)
    report_path = args.out_root / "noisy_seed_metadata.txt"
    report_path.write_text(report)
    print(f"[done F] wrote {report_path}")
    print(report)

    # G: 各色 100 枚 grid 表示
    for color in args.focus_colors:
        grid = build_single_color_grid(samples, color, args.per_grid)
        if grid is None:
            continue
        name = COLOR_NAMES.get(color, str(color))
        out_path = args.out_root / f"noisy_seed_{name}_grid.png"
        cv2.imwrite(str(out_path), grid)
        print(f"[done G] wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
