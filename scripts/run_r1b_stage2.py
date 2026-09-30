"""R1bのON短区間だけを再認識し、固定OFFと原画像で4セルを再検収する。"""
from __future__ import annotations
from pathlib import Path
import shutil
import sys
from scripts import run_r1_stage2 as run, inspect_r1_stage2 as inspect, report_r1_stage2 as report

OUT = Path('logs/r1b/stage2')


def main() -> None:
    """OFFは既存原票を参照コピーし、ONと独立画像観測だけを新しく取得する。"""
    (OUT/'off').mkdir(parents=True, exist_ok=True)
    for name in ('trace.jsonl', 'completed.json'):
        shutil.copy2(Path('logs/r1/stage2/off')/name, OUT/'off'/name)
    run.OUT = inspect.OUT = report.OUT = OUT
    sys.argv = ['r1b_stage2', '--mode', 'on']
    run.main()
    inspect.main()
    report.main()


if __name__ == '__main__':
    main()
