"""タスク#5 (フル物差し回帰と採用登録の準備) の70窓×3構成ジョブ生成。

元ジョブファイル scripts/_jobs_yardstick_v4win_2026-08-06.txt から
(video_stem, out_stem, start_sec, max_sec) の70窓を抽出し、以下3構成の
ジョブファイルを生成する:
    (a) 現行採用構成 (production_config.collect_flags() のみ)
    (b) (a) + --stable-majority-window
    (c) (b) + --enable-ojama-fall-scoped-exit
             + --enable-ojama-fall-placement-override
             + --enable-ojama-fall-entry-hardening

video は $HOME/frames/video_cXX.mp4 (WSL ネイティブ ext4、2026-08-14 に
c10-c23 全14本を再DL/コピー済み) を使う。
"""
from __future__ import annotations

import re
from pathlib import Path

from src.production_config import collect_flags

SRC = Path("scripts/_jobs_yardstick_v4win_2026-08-06.txt")
OUT_DIR_A = "data/verify/board_labels_task5_a_baseline_2026-08-14"
OUT_DIR_B = "data/verify/board_labels_task5_b_smw_2026-08-14"
OUT_DIR_C = "data/verify/board_labels_task5_c_ojamafall_2026-08-14"

LINE_RE = re.compile(
    r"--video \S*/(?P<video>video_c\d+\.mp4).*"
    r"--out-npz \S*/(?P<stem>c\d+_g\d+)\.npz "
    r"--start-sec (?P<start>[0-9.]+) --max-sec (?P<max>[0-9.]+)"
)

FLAGS_B = "--stable-majority-window"
FLAGS_C = (
    "--enable-ojama-fall-scoped-exit "
    "--enable-ojama-fall-placement-override "
    "--enable-ojama-fall-entry-hardening"
)

PY_PREFIX = "PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t"


def parse_windows() -> "list[tuple[str, str, str, str]]":
    """元ジョブファイルから70窓の (video, stem, start_sec, max_sec) を抽出する。"""
    windows: list[tuple[str, str, str, str]] = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # 行頭のタブ区切り番号を除去 (元ファイルは "1\tPYTHONPATH=..." 形式)
        cmd = line.split("\t", 1)[-1]
        m = LINE_RE.search(cmd)
        if not m:
            raise ValueError(f"パース失敗: {line!r}")
        windows.append((m["video"], m["stem"], m["start"], m["max"]))
    return windows


def build_job_line(video: str, stem: str, start: str, max_sec: str, out_dir: str, extra_flags: str) -> str:
    """1窓分のコマンド行を組み立てる (production_config.collect_flags() が単一情報源)。"""
    base_flags = collect_flags()
    flags = f"{base_flags} {extra_flags}".strip()
    return (
        f"{PY_PREFIX} --video $HOME/frames/{video} "
        f"--out-npz {out_dir}/{stem}.npz "
        f"--start-sec {start} --max-sec {max_sec} --sample-interval 0 {flags}"
    )


def main() -> None:
    windows = parse_windows()
    print(f"[gen] 抽出窓数: {len(windows)}")
    configs = (
        ("a_baseline", OUT_DIR_A, ""),
        ("b_smw", OUT_DIR_B, FLAGS_B),
        ("c_ojamafall", OUT_DIR_C, f"{FLAGS_B} {FLAGS_C}"),
    )
    for tag, out_dir, extra in configs:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(f"scripts/_jobs_task5_{tag}_2026-08-14.txt")
        lines = [
            build_job_line(video, stem, start, max_sec, out_dir, extra)
            for video, stem, start, max_sec in windows
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[gen] {out_path} ({len(lines)}行)")


if __name__ == "__main__":
    main()
