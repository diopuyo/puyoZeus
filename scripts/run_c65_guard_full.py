"""cycle65 対整合ガード ON で認識記録を再生成し、E36b と同じ門で本番撃ち合い評価を採点する。

段: collect (認識を ON で再収集→記録) → replay (E36b 構成で再生) → report (E36b 門で採点)。
R1b 収集 (scripts/run_r1b.py) と同じ収集器を使い、コマンドに --next-recolor-pair-guard を
足すだけ。出力は logs/c65_guard/full と logs/c65_guard/e36b_on に隔離し、既存 logs は不変。
使い方: python -m scripts.run_c65_guard_full --step collect|replay|report [--source S]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

GUARD_FLAG = '--next-recolor-pair-guard'
COLLECT_OUT = Path('logs/c65_guard/full')
REPLAY_OUT = Path('logs/c65_guard/e36b_on')
STAGE2_SUMMARY = Path('logs/r1b/stage2/SUMMARY.json')  # R1b の署名検収 (収集器の前提)
SOURCES = ('q_7gc4TgFig', 'review', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')


def collect(source: str) -> None:
    """ガードONで 1 記録を再収集する (R1b の collect と同一手順)。"""
    assert json.loads(STAGE2_SUMMARY.read_text())['passed']
    from scripts import collect_e34c as checked
    from scripts import collect_r1 as capture
    original = checked.command
    checked.command = lambda name: original(name) + [GUARD_FLAG]
    capture.OUT = COLLECT_OUT
    capture.collect(source, 'on')


def replay(source: str) -> None:
    """E36b と同じ構成で、ON 再収集した記録を再生する。"""
    from scripts import run_e36, run_e36b
    run_e36.OUT, run_e36.options = REPLAY_OUT, run_e36b.options
    run_e36.RECORDS = COLLECT_OUT / 'records'
    run_e36.worker(source)


def report() -> None:
    """E36b の採点器を出力先だけ差し替えて実行する。"""
    from scripts import report_e36b
    report_e36b.OUT = REPLAY_OUT
    report_e36b.report()


def spawn(step: str, source: str) -> None:
    """1 記録 1 プロセスで起動する (ログは出力先直下)。"""
    root = COLLECT_OUT if step == 'collect' else REPLAY_OUT
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'{step}_{source}.log').open('a') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'scripts.run_c65_guard_full',
                        '--step', step, '--source', source],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=('collect', 'replay', 'report'), required=True)
    parser.add_argument('--source', choices=SOURCES)
    args = parser.parse_args()
    if args.step == 'report':
        report()
    elif args.source:
        (collect if args.step == 'collect' else replay)(args.source)
    else:
        for source in SOURCES:
            spawn(args.step, source)


if __name__ == '__main__':
    main()
