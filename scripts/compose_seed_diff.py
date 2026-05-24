"""改修前 vs 改修後 の seed PNG を side-by-side 合成 (= ユーザー目視時間短縮)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True, help="改修前 PNG")
    p.add_argument("--after", type=Path, required=True, help="改修後 PNG")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", type=str, default="", help="上部 label (= 動画名)")
    args = p.parse_args()

    if not args.before.exists():
        print(f"[error] before not found: {args.before}", file=sys.stderr)
        return 1
    if not args.after.exists():
        print(f"[error] after not found: {args.after}", file=sys.stderr)
        return 1

    before = cv2.imread(str(args.before))
    after = cv2.imread(str(args.after))
    if before is None or after is None:
        print(f"[error] failed to read", file=sys.stderr)
        return 1

    # 幅を最大に揃える
    max_w = max(before.shape[1], after.shape[1])
    if before.shape[1] != max_w:
        before = cv2.copyMakeBorder(
            before, 0, 0, 0, max_w - before.shape[1], cv2.BORDER_CONSTANT,
        )
    if after.shape[1] != max_w:
        after = cv2.copyMakeBorder(
            after, 0, 0, 0, max_w - after.shape[1], cv2.BORDER_CONSTANT,
        )

    # 各 image にラベル strip 追加 (上に「BEFORE」 / 「AFTER」)
    def add_strip(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
        strip = np.full((30, img.shape[1], 3), 32, dtype=np.uint8)
        cv2.putText(
            strip, text, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )
        return np.vstack([strip, img])

    before_s = add_strip(before, "BEFORE (= cycle 32 系 baseline)", (200, 200, 255))
    after_s = add_strip(after, "AFTER  (= cycle 50 改修 2/3/4)", (200, 255, 200))

    # 上下結合 + 動画ラベル
    combined = np.vstack([before_s, after_s])
    if args.label:
        label_strip = np.full((30, combined.shape[1], 3), 0, dtype=np.uint8)
        cv2.putText(
            label_strip, args.label, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2,
        )
        combined = np.vstack([label_strip, combined])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), combined)
    print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
