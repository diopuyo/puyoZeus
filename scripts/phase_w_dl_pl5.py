"""W: 新規再生リスト (pl5) の動画を順次 DL。

各動画を data/frames/video_NN.mp4 として保存 (NN は既存 video_03 の続き)。
720p 優先 (容量節約)、ダウンロード後に matches/winners 検出を別スクリプトで実行。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_dl_pl5
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
    "https://www.youtube.com/playlist?list=PLsjREVssD8bZcS3TY7BYUOwywpJPuJTh2"
)
# Index 1 (3試合分の長尺) を含む 6 動画
START_VIDEO_INDEX = 4  # video_04 から命名
WORK_DIR = Path("data/frames")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-index", type=int, default=START_VIDEO_INDEX,
        help="保存ファイル名の開始番号 (video_NN.mp4 の NN)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="ダウンロード解像度上限 (default 720)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最大何動画 DL するか (0=全部)",
    )
    args = parser.parse_args()

    yt_dlp = str(_ROOT / "venv/bin/yt-dlp")

    # プレイリスト URL 取得
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
            print(f"[{i+1}/{len(items)}] video_{out_idx:02d}.mp4 SKIP (already DL)")
            continue
        # part ファイル削除
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
    for p in sorted(WORK_DIR.glob("video_*.mp4")):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {to_windows_path(p)}: {size_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
