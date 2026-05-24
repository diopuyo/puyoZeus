"""Phase E-2 補助: 複数 playlist を順次 DL (video_NN.mp4 連番).

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_dl_playlists \
        --start-index 20 --max-per-playlist 6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


# ユーザー指定 6 プレイリストのうち、新規 + 再取得対象 4 つ。
# pl1 (PLsjREVssD8baPrsapHGszFzhObLANmd6B) は video_01-03 として既存。
# pl6 (PLsjREVssD8bY6jUJbp7CYZT8pS2JBBt6C) は video_10-19 として既存。
_ALL_PLAYLISTS: tuple[tuple[str, str], ...] = (
    ("plA", "https://www.youtube.com/playlist?"
            "list=PLsjREVssD8bZ0_gTx4r2S2MeoG65Trubl"),
    ("plB", "https://www.youtube.com/playlist?"
            "list=PLsjREVssD8baOyWw8zpRqV0ru42Cy52Ik"),
    ("plC", "https://www.youtube.com/playlist?"
            "list=PLsjREVssD8bZer2yBUdJ9ZPrvJ0SeLJi8"),
    ("plD", "https://www.youtube.com/playlist?"
            "list=PLsjREVssD8bYG_VUIlJvREnco92HB5R3t"),
)
WORK_DIR = Path("data/frames")
INDEX_LOG = Path("data/phase_e_dl_index.tsv")


def fetch_playlist_items(yt_dlp: str, url: str) -> list[tuple[int, str, int, str]]:
    r = subprocess.run(
        [yt_dlp, "--flat-playlist",
         "--print", "%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s",
         url],
        capture_output=True, text=True, check=True,
    )
    items = []
    for line in r.stdout.strip().split("\n"):
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        idx, vid_id, dur, title = parts
        try:
            dur_i = int(float(dur)) if dur != "NA" else 0
            idx_i = int(float(idx))
        except ValueError:
            continue
        items.append((idx_i, vid_id, dur_i, title))
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-index", type=int, default=20,
        help="保存ファイル名の開始番号 (video_NN.mp4 の NN)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="DL 解像度上限",
    )
    parser.add_argument(
        "--max-per-playlist", type=int, default=0,
        help="各 playlist あたり最大 DL 数 (0=全て)",
    )
    parser.add_argument(
        "--min-duration", type=int, default=300,
        help="この秒数未満の動画は skip (短い切り抜きは除外)",
    )
    parser.add_argument(
        "--playlists", type=str, default="",
        help="対象プレイリストタグ (例: plC,plD). 空=全部",
    )
    parser.add_argument(
        "--skip-playlist-idx", type=str, default="",
        help="プレイリスト idx (1-origin) を range で skip. 例: 'plB:1-6,plC:1-6,plD:1-6' で各先頭 6 本を skip",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="並列 DL 数 (>=2 で multiprocessing による並列実行)",
    )
    args = parser.parse_args()
    if args.playlists:
        wanted = {s.strip() for s in args.playlists.split(",") if s.strip()}
        playlists = tuple(p for p in _ALL_PLAYLISTS if p[0] in wanted)
    else:
        playlists = _ALL_PLAYLISTS
    skip_map: dict[str, set[int]] = {}
    if args.skip_playlist_idx:
        for chunk in args.skip_playlist_idx.split(","):
            chunk = chunk.strip()
            if ":" not in chunk:
                continue
            tag, rng = chunk.split(":", 1)
            tag = tag.strip()
            if "-" in rng:
                a, b = rng.split("-", 1)
                skip_map.setdefault(tag, set()).update(
                    range(int(a), int(b) + 1),
                )
            else:
                skip_map.setdefault(tag, set()).add(int(rng))

    yt_dlp = str(_ROOT / "venv" / "bin" / "yt-dlp")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_LOG.parent.mkdir(parents=True, exist_ok=True)

    next_idx = args.start_index
    log_lines: list[str] = []
    if not INDEX_LOG.exists():
        log_lines.append(
            "video_idx\tplaylist\tplaylist_idx\tvideo_id\tduration\ttitle",
        )

    # まず全 DL タスクを構築 (並列でも逐次でも統一処理)
    tasks: list[tuple[str, int, str, int, str, int]] = []
    # (tag, pl_idx, vid_id, dur, title, next_idx)
    for tag, url in playlists:
        print(f"\n=== {tag}: {url}")
        try:
            items = fetch_playlist_items(yt_dlp, url)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED to fetch playlist: {e}")
            continue
        print(f"  playlist has {len(items)} videos")
        if args.max_per_playlist > 0:
            items = items[:args.max_per_playlist]

        for pl_idx, vid_id, dur, title in items:
            if pl_idx in skip_map.get(tag, set()):
                print(f"  [skip-idx] {tag} idx={pl_idx} {vid_id}")
                continue
            if dur < args.min_duration:
                print(
                    f"  [skip] short ({dur}s) {vid_id} {title[:40]}"
                )
                continue
            out_path = WORK_DIR / f"video_{next_idx:02d}.mp4"
            if out_path.exists() and out_path.stat().st_size > 10_000_000:
                print(
                    f"  [skip] video_{next_idx:02d}.mp4 already exists"
                )
                next_idx += 1
                continue
            tasks.append((tag, pl_idx, vid_id, dur, title, next_idx))
            next_idx += 1

    print(f"\n[plan] {len(tasks)} videos to DL "
          f"(parallel={args.parallel})")

    def _dl_one(t: tuple) -> tuple[bool, int, str, int, str, int, str]:
        tag, pl_idx, vid_id, dur, title, idx = t
        out_path = WORK_DIR / f"video_{idx:02d}.mp4"
        part = Path(str(out_path) + ".part")
        part.unlink(missing_ok=True)
        cmd = [
            yt_dlp, "-f",
            f"bestvideo[ext=mp4][vcodec^=avc1][height<={args.height}]/"
            f"bestvideo[ext=mp4][height<={args.height}]",
            "-o", str(out_path), "--no-playlist", "--quiet",
            f"https://www.youtube.com/watch?v={vid_id}",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as e:
            return (False, idx, vid_id, dur, title, pl_idx, tag)
        return (True, idx, vid_id, dur, title, pl_idx, tag)

    if args.parallel >= 2:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_dl_one, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                ok, idx, vid_id, dur, title, pl_idx, tag = fut.result()
                done += 1
                if ok:
                    out_path = WORK_DIR / f"video_{idx:02d}.mp4"
                    size_mb = (
                        out_path.stat().st_size / 1024 / 1024
                        if out_path.exists() else 0
                    )
                    print(
                        f"  [{done}/{len(tasks)}] DL ok: "
                        f"{tag} idx={pl_idx} -> video_{idx:02d}.mp4 "
                        f"({size_mb:.0f} MB)"
                    )
                    log_lines.append(
                        f"{idx:02d}\t{tag}\t{pl_idx}\t{vid_id}\t{dur}\t{title}"
                    )
                else:
                    print(f"  [FAIL] {tag} idx={pl_idx} {vid_id}")
    else:
        for t in tasks:
            ok, idx, vid_id, dur, title, pl_idx, tag = _dl_one(t)
            if ok:
                out_path = WORK_DIR / f"video_{idx:02d}.mp4"
                size_mb = (
                    out_path.stat().st_size / 1024 / 1024
                    if out_path.exists() else 0
                )
                print(
                    f"  [{tag} pl_idx={pl_idx}] saved "
                    f"video_{idx:02d}.mp4 ({size_mb:.0f} MB)"
                )
                log_lines.append(
                    f"{idx:02d}\t{tag}\t{pl_idx}\t{vid_id}\t{dur}\t{title}"
                )
            else:
                print(f"    FAILED: {vid_id}")

    if log_lines:
        mode = "a" if INDEX_LOG.exists() else "w"
        with INDEX_LOG.open(mode, encoding="utf-8") as f:
            for ln in log_lines:
                f.write(ln + "\n")
        print(f"\n[log] {to_windows_path(INDEX_LOG)}")

    print("\n=== complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
