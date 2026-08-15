"""指摘16調査: 画像の任意矩形をクロップして拡大保存する (計装専用)。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"read失敗: {args.image}")
    crop = img[args.y:args.y + args.h, args.x:args.x + args.w]
    if args.scale != 1.0:
        crop = cv2.resize(
            crop, None, fx=args.scale, fy=args.scale,
            interpolation=cv2.INTER_NEAREST,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), crop)
    print(f"saved {args.out} shape={crop.shape}")


if __name__ == "__main__":
    main()
