"""D1〜D4: 元のコマンド・収集器を保ち、追加フラグと出力先だけ変更する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._diag_set2_falsedeath_batch import ROOT, EXEV, SOURCES, records

OUT = ROOT / 'logs/eval_set/falsedeath_fix'
FLAGS = ('--death-input-integrity', '--post-counter-input-integrity',
         '--score-attribution', '--falling-puyo-integrity')


def original_command(name: str) -> list[str]:
    """既存の完了票から実際に使った区間・全引数を読む。"""
    path = records()[name]
    candidates = (path.with_suffix('.json'), path.with_suffix('').with_suffix('.jsonl.json'))
    for candidate in candidates:
        if candidate.exists():
            value = json.loads(candidate.read_text())
            if 'command' in value:
                return value['command']
    part = ('set2/collect' if name.startswith('s') else 'collect')
    return json.loads((ROOT / 'logs/eval_set' / part / 'capture' / name / 'COMMAND.json').read_text())['args']


def command(name: str) -> list[str]:
    """R1自身が加える一フラグを除き、元の引数を維持する。"""
    from scripts.collect_e34c import OUTPUT_FLAGS
    args = [v for v in original_command(name) if v != '--placement-signal-reconcile']
    for flag in OUTPUT_FLAGS:
        if flag in args:
            index = args.index(flag) + 1
            args[index] = str(OUT / 'collect/capture' / name / Path(args[index]).name)
    return args + list(FLAGS)


def main() -> None:
    """既存R1収集器を一記録だけ実行し、上書きを拒否する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=tuple(records()))
    args = parser.parse_args()
    from scripts import collect_e34b as capture, collect_e34c as checked, collect_r1
    from scripts import e34c_observations
    assert not (OUT / 'collect/completed' / f'{args.name}.json').exists()
    capture.ROOT = EXEV
    checked.command = command
    before = checked.has_before_board
    checked.has_before_board = lambda name: before(name if name in SOURCES else 'zenchi')
    collect_r1.fire_keys = e34c_observations.fire_keys = lambda name: set()
    collect_r1.OUT = OUT / 'collect'
    collect_r1.collect(args.name, 'on')


if __name__ == '__main__':
    main()
