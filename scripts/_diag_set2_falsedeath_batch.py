"""診断再生を逐次実行するワーカー。複数起動時は外部で最大4本に制限する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/mnt/d/puyo_analyzer/wt_evalset')
OUT = ROOT / 'logs/eval_set/set2_falsedeath'
EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
SOURCES = ('review', 'q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')


def records() -> dict[str, Path]:
    """既存5記録・セット1・セット2の全区間を列挙する。"""
    result = {s: EXEV / 'logs/pending_expiry/full/records' / f'{s}.jsonl.gz' for s in SOURCES}
    for prefix, directory, count in (('c', 'collect', 6), ('s', 'set2/collect', 7)):
        for index in range(1 if prefix == 'c' else 0, count + (prefix == 'c')):
            name = f'{prefix}{index}'
            result[name] = ROOT / 'logs/eval_set' / directory / 'records' / f'{name}.jsonl.gz'
    return result


def main() -> None:
    """execで次の独立Pythonへ置換し、待機用の追加プロセスを作らない。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--mode', default='combined')
    args = parser.parse_args()
    from scripts._diag_set2_falsedeath_extra import main as extra
    extra()
    from scripts._diag_set2_falsedeath_local import main as local
    local()
    from scripts._diag_set2_falsedeath_audit import prepare_baselines
    prepare_baselines()
    pending = [n for n in args.names if not (OUT / args.mode / n / 'status.json').exists()]
    if not pending:
        print('全指定記録完了', flush=True)
        return
    name = next((n for n in pending if claim(args.mode, n)), None)
    if name is None:
        print('残区間は他の診断ワーカーが担当中', flush=True)
        return
    pending = pending[pending.index(name):]
    from scripts._diag_set2_falsedeath_replay import main as replay
    sys.argv = ['replay', str(records()[name]), name, '--mode', args.mode]
    replay()
    receipt = OUT / f'batch_{args.mode}_{name}.json'
    receipt.write_text(json.dumps(dict(completed=name, remaining=pending[1:])), encoding='utf-8')
    if len(pending) > 1:
        os.execv(sys.executable, [sys.executable, '-B', '-m',
                 'scripts._diag_set2_falsedeath_batch', '--mode', args.mode, *pending[1:]])


def claim(mode: str, name: str) -> bool:
    """同じ出力への並行書込みを原子的なディレクトリ作成で防ぐ。"""
    directory = OUT / 'claims' / mode
    directory.mkdir(parents=True, exist_ok=True)
    try:
        (directory / name).mkdir()
        return True
    except FileExistsError:
        return False


if __name__ == '__main__':
    main()
