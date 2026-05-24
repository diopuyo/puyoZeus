"""2 サイクルの viz mp4 を side-by-side 比較動画に合成 (ffmpeg).

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts.cycle_compare_video \
        --left  data/test_unknown/v97_match11_96s_viz_cycle_0.mp4 \
        --right data/test_unknown/v97_match11_96s_viz_cycle_2.mp4 \
        --output data/test_unknown/compare_cycle_0_vs_2_v97.mp4

縮小なし (= 1920x1080 + 1920x1080 → 3840x1080 widescreen) で並べる。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True, type=Path)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scale", type=float, default=0.5)
    args = p.parse_args()

    if not args.left.exists() or not args.right.exists():
        print(f"[error] missing input(s)", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _get_ffmpeg()
    # scale で縮小して horizontal stack. drawtext filter は imageio_ffmpeg
    # ビルドに非搭載なため省略 (= 単純な scale + hstack).
    s = args.scale
    cmd = [
        ffmpeg, "-y",
        "-i", str(args.left),
        "-i", str(args.right),
        "-filter_complex",
        (
            f"[0:v]scale=iw*{s}:ih*{s}[l];"
            f"[1:v]scale=iw*{s}:ih*{s}[r];"
            f"[l][r]hstack=inputs=2"
        ),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an",
        str(args.output),
    ]
    print(f"[ffmpeg] {' '.join(cmd[:6])} ... -> {args.output}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] ffmpeg failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
