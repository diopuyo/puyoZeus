"""Milestone ポーラー。未読エントリを標準出力に、最終既読を data/milestone_last_seen に更新する。

使い方:
    python scripts/poll_milestones.py

出力 (標準出力):
    TOTAL=<n> LAST=<m>
    NEW=<n-m>   # 未読件数
    [未読 JSON 行]
    PROC_STATUS=<watchdog_pid>|<wrapper>|<python>
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MILESTONES = ROOT / "data" / "milestones.jsonl"
LAST_SEEN = ROOT / "data" / "milestone_last_seen"
WATCHDOG_PID = ROOT / "data" / "watchdog.pid"


def main() -> None:
    total = 0
    if MILESTONES.exists():
        with open(MILESTONES, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

    last = 0
    if LAST_SEEN.exists():
        try:
            last = int(LAST_SEEN.read_text().strip() or "0")
        except ValueError:
            last = 0

    print(f"TOTAL={total} LAST={last}")
    new_n = max(0, total - last)
    print(f"NEW={new_n}")

    if new_n > 0:
        with open(MILESTONES, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i > last:
                    print(line.rstrip())
        LAST_SEEN.write_text(str(total))

    # プロセス生存
    wd_pid = WATCHDOG_PID.read_text().strip() if WATCHDOG_PID.exists() else "?"
    try:
        wrap = subprocess.check_output(["pgrep", "-af", "run_long_improve_wrapper"], text=True).strip()
    except subprocess.CalledProcessError:
        wrap = ""
    try:
        py = subprocess.check_output(["pgrep", "-af", "long_improve_v2"], text=True).strip()
    except subprocess.CalledProcessError:
        py = ""
    wrap_pid = wrap.split()[0] if wrap else "DEAD"
    py_pid = py.split()[0] if py else "DEAD"
    print(f"PROC_STATUS=watchdog={wd_pid} wrapper={wrap_pid} python={py_pid}")


if __name__ == "__main__":
    main()
