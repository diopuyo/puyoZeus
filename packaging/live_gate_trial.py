"""配布21cを変更せず、測定するプロセス木だけにCPU条件を付ける。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_gate_win import attach_job
from run_obs_abba_trial import idle_once
import smoke_live_obs_20261020 as smoke


def main() -> None:
    if os.environ.get('LIVE_GATE_AUTHORIZED') != 'ABBA-21C-FULL':
        raise RuntimeError('単独起動は禁止。門番付きcmdから実行する')
    observed = attach_job(int(os.environ['LIVE_GATE_MASK']), os.environ['LIVE_GATE_JOB'])
    Path(os.environ['LIVE_GATE_CPU_RECEIPT']).write_text(
        json.dumps(observed, indent=2), encoding='utf-8')
    smoke.guard = idle_once
    smoke.run_cli()


if __name__ == '__main__':
    main()
