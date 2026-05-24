"""W: pl6 再生リスト (マスター・Bブロック等) の動画を順次 DL。

pl5 と並列実行可能。video_10 から命名。
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


PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=PLsjREVssD8bY6jUJbp7CYZT8pS2JBBt6C"
)
START_VIDEO_INDEX = 10
WORK_DIR = Path("data/frames")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=START_VIDEO_INDEX)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--limit", type=int, default=10,
        help="最大 DL 動画数 (default 10)",
    )
    args = parser.parse_args()

    yt_dlp = str(_ROOT / "venv/bin/yt-dlp")
    print(f"fetching playlist: {PLAYLIST_URL}")
    r = subprocess.run(
        [yt_dlp, "--flat-playlist",
         "--print", "%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s",
         PLAYLIST_URL],
        capture_output=True, text=True, check=True,
    )
    items = []
    for line in r.stdout.strip().split("\n"):
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        idx, vid_id, dur, title = parts
        items.append((int(idx), vid_id, int(dur) if dur != "NA" else 0, title))
    print(f"playlist has {len(items)} videos")

    if args.limit > 0:
        items = items[:args.limit]

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    for i, (pl_idx, vid_id, dur, title) in enumerate(items):
        out_idx = args.start_index + i
        out_path = WORK_DIR / f"video_{out_idx:02d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 10_000_000:
            print(f"[{i+1}/{len(items)}] video_{out_idx:02d}.mp4 SKIP")
            continue
        part = Path(str(out_path) + ".part")
        part.unlink(missing_ok=True)

        print(
            f"[{i+1}/{len(items)}] DL idx={pl_idx} id={vid_id} "
            f"({dur}s) -> video_{out_idx:02d}.mp4"
        )
        cmd = [
            yt_dlp, "-f",
            f"bestvideo[ext=mp4][vcodec^=avc1][height<={args.height}]/"
            f"bestvideo[ext=mp4][height<={args.height}]",
            "-o", str(out_path), "--no-playlist", "--quiet",
            f"https://www.youtube.com/watch?v={vid_id}",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  FAILED: {e}")
            continue
        size_mb = out_path.stat().st_size / 1024 / 1024 if out_path.exists() else 0
        print(f"  saved {size_mb:.0f} MB")

    print("\n=== complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
