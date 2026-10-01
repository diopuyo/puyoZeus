"""評価セット: 再判定の10モデル (旧/補正 × random_state 0〜4) を3パートの記録で再生する (2026-10-01)。

wt_nextfix の再生器 (scripts/next_shift_replay_20261001.run) をそのまま使い、出力先と記録の
置き場所だけ差し替える。実行は PYTHONPATH=/mnt/d/puyo_analyzer/wt_nextfix・cwd=/mnt/d/puyo_analyzer/nextfix_run
(q .505〜.518 を出した再生と同じコード・同じ cwd)。本ファイルは wt_evalset に置き、ファイルパスで起動する。
使い方: python <本ファイル> --variant J_orig_rs0 --source p1|p2|zenchi
"""
from __future__ import annotations

import argparse
from pathlib import Path

EVALSET = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set')
OUT_ROOT = EVALSET/'replay_rejudge'
PART_RECORDS = EVALSET/'collect/records'
PART3_RECORDS = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/full/records')
SOURCES = ('p1', 'p2', 'zenchi')


def main() -> None:
    """1構成×1記録を1プロセスで再生する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--source', choices=SOURCES, required=True)
    args = parser.parse_args()
    from scripts import next_shift_replay_20261001 as base
    assert Path(base.__file__).resolve().is_relative_to(Path('/mnt/d/puyo_analyzer/wt_nextfix'))
    base.OUT_ROOT = OUT_ROOT
    base.RECORDS = PART3_RECORDS if args.source == 'zenchi' else PART_RECORDS
    base.run(args.variant, args.source)


if __name__ == '__main__':
    main()
