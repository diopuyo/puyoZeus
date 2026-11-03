"""本番・全修正・一つ抜きの再生。モデル・本番設定は共通CLIから取得する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Iterator
from types import ModuleType

from scripts.falsedeath_collect_20261001 import OUT, FLAGS, records

MODES = ('prod', 'all', 'no_d1', 'no_d2', 'no_d3', 'no_d4', 'v2', 'v2b', 'v2c', 'v2d', 'v2e', 'v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b')
V2_FLAGS = ('--death-input-integrity', '--post-counter-input-integrity-v2',
            '--score-attribution-v2', '--post-counter-current-puyo-integrity')


def enabled_flags(mode: str) -> list[str]:
    """旧構成の並びを保ち、D-v2だけ事前登録のフラグへ置き換える。"""
    if mode == 'v2h2b':
        return enabled_flags('v2h2') + ['--death-freshness-cap']
    if mode == 'v2h2':
        return enabled_flags('v2h') + ['--death-integrity-v2h2']
    if mode == 'v2h':
        return enabled_flags('v2e') + ['--post-counter-completion-recent-integrity']
    if mode == 'v2g':
        return enabled_flags('v2e') + ['--post-counter-completion-step-integrity']
    if mode == 'v2f':
        return enabled_flags('v2e') + ['--post-counter-completion-prefix-integrity']
    if mode == 'v2e':
        return enabled_flags('v2d') + ['--death-history-boundary-start']
    if mode == 'v2d':
        return enabled_flags('v2c') + ['--death-history-integrity']
    if mode == 'v2':
        return list(V2_FLAGS)
    if mode == 'v2b':
        return [f for f in V2_FLAGS if f != '--post-counter-input-integrity-v2']
    if mode == 'v2c':
        return ['--death-input-integrity', '--score-attribution-v2',
                '--post-counter-current-puyo-integrity-v2c']
    return [] if mode == 'prod' else [f for i, f in enumerate(FLAGS[:3], 1) if mode != f'no_d{i}']


def limit_replay(module: ModuleType, end_sec: float) -> None:
    """記録先頭の履歴を維持して対象直後で止め、既存記録には書き込まない。"""
    original = module.read_records

    def limited(path: Path) -> Iterator[dict]:
        for row in original(path):
            if row['kind'] == 'update' and row['args'][3] > end_sec:
                break
            yield row

    module.read_records = limited


def input_record(mode: str, name: str) -> Path:
    """D4を外すときだけ旧認識記録を使う。"""
    return records()[name] if mode in ('prod', 'no_d4') else OUT / 'collect/records' / f'{name}.jsonl.gz'


def save_v2e_resources(dest: Path) -> None:
    """実計算プロセスのnice・空きメモリ・BLAS制限を起動時に検証する。"""
    import os
    from datetime import datetime, timezone
    minimum_kib = 6 * 1024**2
    pools = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    row = dict(nice=os.getpriority(os.PRIO_PROCESS, 0),
        available_kib=int(memory['MemAvailable'].split()[0]),
        pools={key: os.environ.get(key) for key in pools},
        started_utc=datetime.now(timezone.utc).isoformat())
    assert row['nice'] == 10 and row['available_kib'] >= minimum_kib, row
    assert all(value == '1' for value in row['pools'].values()), row
    with (dest/'RESOURCE.json').open('x', encoding='utf-8') as stream:
        json.dump(row, stream, indent=1)


def main() -> None:
    """一構成・一記録だけを新規出力先へ再生する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('name', choices=tuple(records()))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--end-sec', type=float)
    args = parser.parse_args()
    if args.mode in ('v2', 'v2b', 'v2c', 'v2d', 'v2e', 'v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b') and args.out is None:
        gate = json.loads((OUT/args.mode/'LADDER.json').read_text(encoding='utf-8'))
        assert gate['passed'], '短区間の停止条件により全体再生禁止'
    dest = args.out or OUT / 'replay' / args.mode / args.name
    assert not (dest / 'status.json').exists(), '完成済み成果物は上書きしない'
    dest.mkdir(parents=True, exist_ok=True)
    if args.mode in ('v2e', 'v2f', 'v2g', 'v2h', 'v2h2', 'v2h2b'):
        save_v2e_resources(dest)
    enabled = enabled_flags(args.mode)
    from scripts._diag_set2_falsedeath_replay import install_progress, install_trace
    install_progress()
    install_trace(dest)
    sys.argv = ['replay', str(input_record(args.mode, args.name)), '--out', str(dest),
                '--production-exchange-event', *enabled]
    (dest / 'COMMAND.json').write_text(json.dumps(sys.argv, ensure_ascii=False, indent=1))
    from scripts import replay_exchange_event_20260926 as replay
    if args.end_sec is not None:
        limit_replay(replay, args.end_sec)
    replay.main()


if __name__ == '__main__':
    main()
