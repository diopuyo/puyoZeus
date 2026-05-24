"""Phase Z: cross_video の各動画について violations を抽出して人間レビュー素材を準備。

phase_z_cross_video.py が生成した cross_video/v??_m_*/labels.csv 全てに対し、
phase_z_extract_violations.py 相当の処理を回し、各動画の violations_review/
を準備。ユーザー起床時に即レビュー可能な状態にする。
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

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-dir",
        type=Path,
        default=_ROOT / "data/verify/phase_z_review/cross_video",
    )
    args = parser.parse_args()

    if not args.cross_dir.exists():
        print(f"ERROR: {args.cross_dir} not found")
        return 1

    # cross_video 配下の各 v?? 動画ディレクトリを処理
    video_dirs = sorted(
        d for d in args.cross_dir.iterdir()
        if d.is_dir() and d.name.startswith("v")
    )
    print(f"対象動画: {len(video_dirs)} 個")
    for vdir in video_dirs:
        labels = vdir / "labels.csv"
        if not labels.exists():
            print(f"[skip] {vdir.name}: labels.csv なし")
            continue
        # 動画 id 抽出 (例: "v04_m_1083_1113" → "04")
        parts = vdir.name.split("_")
        if not parts[0].startswith("v"):
            continue
        vid_str = parts[0][1:]
        try:
            vid = int(vid_str)
        except ValueError:
            continue
        video_path = _ROOT / f"data/frames/video_{vid:02d}.mp4"
        if not video_path.exists():
            print(f"[skip] {vdir.name}: 動画ファイルなし")
            continue
        out_dir = vdir / "violations_review"
        if out_dir.exists() and (out_dir / "violations.csv").exists():
            print(f"[done] {vdir.name}: 抽出済 (skip)")
            continue
        cmd = [
            "./venv/bin/python", "-m",
            "scripts.phase_z_extract_violations",
            "--labels", str(labels),
            "--video", str(video_path),
            "--out-dir", str(out_dir),
        ]
        env = {**os.environ, "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}
        print(f"[run] {vdir.name}")
        try:
            result = subprocess.run(
                cmd, cwd=str(_ROOT), env=env,
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                print(f"  FAIL: {result.stderr[-200:]}")
                continue
            # 末尾だけ表示
            for line in result.stdout.strip().splitlines()[-3:]:
                print(f"  {line}")
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")

    print()
    print("全動画 violations 抽出完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
