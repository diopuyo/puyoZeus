"""手動ラベル npz をデータ拡張で 5 倍に増やす (Phase U-C)。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

import cv2
import numpy as np


def aug_flip(p: np.ndarray, rng) -> np.ndarray:
    return cv2.flip(p, 1)


def aug_noise(p: np.ndarray, rng) -> np.ndarray:
    noise = rng.normal(0, 5, p.shape).astype(np.int16)
    return np.clip(p.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def aug_hsv(p: np.ndarray, rng) -> np.ndarray:
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-3, 4))) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + int(rng.integers(-15, 16)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + int(rng.integers(-15, 16)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def aug_rotate(p: np.ndarray, rng) -> np.ndarray:
    h, w = p.shape[:2]
    angle = float(rng.uniform(-5, 5))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(p, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


AUGS = (aug_flip, aug_noise, aug_hsv, aug_rotate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/training_phase_u/manual_labels.npz",
    )
    parser.add_argument(
        "--output", default="data/training_phase_u/manual_labels_aug.npz",
    )
    parser.add_argument("--multiplier", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.input)
    patches = data["patches"]
    labels = data["labels"]
    rng = np.random.default_rng(42)

    out_p = []
    out_l = []
    for p, l in zip(patches, labels):
        out_p.append(p)
        out_l.append(int(l))
        for _ in range(max(0, args.multiplier - 1)):
            fn = AUGS[int(rng.integers(0, len(AUGS)))]
            out_p.append(fn(p, rng))
            out_l.append(int(l))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        patches=np.array(out_p, dtype=np.uint8),
        labels=np.array(out_l, dtype=np.int32),
    )
    print(f"saved: {args.output} ({len(out_p)} samples, x{args.multiplier})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
