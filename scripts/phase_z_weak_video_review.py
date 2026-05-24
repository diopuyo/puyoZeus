"""弱点動画 (v04/v06/v12/v16/v19) の追加 review 区間生成。

cross_video summary で精度低めの動画について、複数試合の 30s 区間で
review シートを生成。ユーザーが review labels を入力後、CNN v18 訓練
データに統合できる。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_weak_video_review \
        --videos 4,6,12,16,19 --matches 3,4,5
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def get_match(video_id: int, idx: int) -> tuple[float, float] | None:
    candidates = [
        _ROOT
        / f"data/verify/match_boundaries_v5/video_{video_id:02d}/matches.tsv",
        _ROOT
        / f"data/verify/match_boundaries_v4/video_{video_id:02d}/matches.tsv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
            if idx - 1 < len(rows):
                r = rows[idx - 1]
                return (float(r["start_sec"]), float(r["end_sec"]))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos", default="4,6,12,16,19",
        help="弱点動画 ID (カンマ区切り)",
    )
    parser.add_argument(
        "--matches", default="3,4,5",
        help="試合 idx (カンマ区切り)",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    out_root = _ROOT / "data/verify/phase_z_review/weak_video_extra"
    out_root.mkdir(parents=True, exist_ok=True)

    video_ids = [int(s) for s in args.videos.split(",")]
    match_idxs = [int(s) for s in args.matches.split(",")]
    print(f"対象動画: {len(video_ids)} × 試合 {len(match_idxs)} = "
          f"{len(video_ids) * len(match_idxs)} 区間")
    for vid in video_ids:
        for idx in match_idxs:
            match = get_match(vid, idx)
            if match is None:
                print(f"[skip] v{vid:02d} match {idx}: 試合区間なし")
                continue
            match_start, match_end = match
            start = match_start
            end = min(match_start + args.duration, match_end)
            if end - start < 10:
                print(f"[skip] v{vid:02d} m{idx}: 試合短すぎ {end-start:.0f}s")
                continue
            out_dir = (
                out_root
                / f"v{vid:02d}_m{idx:02d}_{int(start)}_{int(end)}"
            )
            if (out_dir / "violations.csv").exists():
                print(f"[done] {out_dir.name}: 抽出済 (skip)")
                continue
            print(f"[run] v{vid:02d} m{idx} {start:.0f}-{end:.0f}s")
            # phase_z_review_ui で labels.csv 生成
            cmd1 = [
                "./venv/bin/python", "-m", "scripts.phase_z_review_ui",
                "--video", f"data/frames/video_{vid:02d}.mp4",
                "--start", str(start), "--end", str(end),
                "--bg-fp-time", str(start),
                "--out-dir", str(out_dir),
            ]
            env = {**os.environ, "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}
            try:
                r1 = subprocess.run(
                    cmd1, cwd=str(_ROOT), env=env,
                    capture_output=True, text=True, timeout=600,
                )
                if r1.returncode != 0:
                    print(f"  review_ui FAIL: {r1.stderr[-200:]}")
                    continue
            except subprocess.TimeoutExpired:
                print("  review_ui TIMEOUT")
                continue
            # violations 抽出
            violations_dir = out_dir / "violations_review"
            cmd2 = [
                "./venv/bin/python", "-m",
                "scripts.phase_z_extract_violations",
                "--labels", str(out_dir / "labels.csv"),
                "--video", f"data/frames/video_{vid:02d}.mp4",
                "--out-dir", str(violations_dir),
            ]
            try:
                r2 = subprocess.run(
                    cmd2, cwd=str(_ROOT), env=env,
                    capture_output=True, text=True, timeout=300,
                )
                if r2.returncode != 0:
                    print(f"  violations FAIL: {r2.stderr[-200:]}")
                    continue
                for line in r2.stdout.splitlines()[-3:]:
                    print(f"  {line}")
            except subprocess.TimeoutExpired:
                print("  violations TIMEOUT")
    print("\n全弱点動画区間の review シート生成完了")
    print(f"出力: {to_windows_path(out_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
