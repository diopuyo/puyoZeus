"""env propagation 修正後の真 threshold sweep。

cross_video を複数 PHASE_Z_* env var 設定で順次回し summary を生成。
v16_clean baseline と比較する用途。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_sweep_thresholds \
        --variants emS50,vS80,tv2

variant ID:
    emS40 / emS50 / emS70 / emS80         (PHASE_Z_EM_S_MIN sweep)
    vS80 / vS90 / vS110 / vS130           (PHASE_Z_HSV_VOTE_S_MIN sweep)
    tv2 / tv4 / tv5                       (PHASE_Z_TV_WINDOW sweep)

各 variant の cross_video summary は data/verify/phase_z_review/cross_video_v16_<id>/
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

VARIANT_ENV: dict[str, dict[str, str]] = {
    "emS40": {"PHASE_Z_EM_S_MIN": "40"},
    "emS50": {"PHASE_Z_EM_S_MIN": "50"},
    "emS70": {"PHASE_Z_EM_S_MIN": "70"},
    "emS80": {"PHASE_Z_EM_S_MIN": "80"},
    "vS80":  {"PHASE_Z_HSV_VOTE_S_MIN": "80"},
    "vS90":  {"PHASE_Z_HSV_VOTE_S_MIN": "90"},
    "vS110": {"PHASE_Z_HSV_VOTE_S_MIN": "110"},
    "vS130": {"PHASE_Z_HSV_VOTE_S_MIN": "130"},
    "tv2":   {"PHASE_Z_TV_WINDOW": "2"},
    "tv4":   {"PHASE_Z_TV_WINDOW": "4"},
    "tv5":   {"PHASE_Z_TV_WINDOW": "5"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", default="emS50,vS80,tv2",
        help=f"カンマ区切りで variant ID (利用可能: {','.join(VARIANT_ENV)})",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--videos",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19",
    )
    args = parser.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    for v in variants:
        if v not in VARIANT_ENV:
            print(f"ERROR: unknown variant {v}")
            return 1

    for v in variants:
        env_vars = VARIANT_ENV[v]
        out_suffix = f"v16_{v}_clean"
        out_dir = (
            _ROOT / "data/verify/phase_z_review"
            / f"cross_video_{out_suffix}"
        )
        if (out_dir / "summary.tsv").exists():
            print(f"[skip] {v}: 既に summary.tsv 存在 ({out_dir})")
            continue
        print()
        print("=" * 60)
        print(f"sweep variant: {v}")
        for k, val in env_vars.items():
            print(f"  {k}={val}")
        print("=" * 60)
        cmd = [
            "./venv/bin/python", "-m", "scripts.phase_z_cross_video",
            "--videos", args.videos,
            "--duration", str(args.duration),
            "--out-suffix", out_suffix,
        ]
        env = {**os.environ, **env_vars, "PYTHONPATH": "."}
        result = subprocess.run(
            cmd, cwd=str(_ROOT), env=env, check=False,
        )
        if result.returncode != 0:
            print(f"[fail] variant {v}: returncode={result.returncode}")
        else:
            print(f"[done] variant {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
