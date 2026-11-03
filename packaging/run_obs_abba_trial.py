"""ABBA各試行の開始前負荷を一度確認し、忙しければ正式測定を拒否する。"""
from pathlib import Path
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_live_obs_20261020 as smoke


def idle_once(benchmark: bool, out: Path, wait_started_at: str | None = None) -> None:
    if not benchmark:
        raise ValueError('ABBAはbenchmark指定が必須')
    subprocess.run(['powershell', '-NoProfile', '-File',
        str(Path(__file__).with_name('wait_live_obs_idle.ps1')),
        '-Output', str(out/'idle_checks.jsonl'), '-ExcludePid', str(os.getpid()),
        '-SingleCheck'], check=True)
    last = json.loads((out/'idle_checks.jsonl').read_text(encoding='utf-8-sig').splitlines()[-1])
    if last['busy']:
        raise RuntimeError('同居する重い計算を検出。正式ABBAの開始を保留')


if __name__ == '__main__':
    smoke.guard = idle_once
    smoke.run_cli()
