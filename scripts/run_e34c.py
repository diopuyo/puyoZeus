"""E34cの収集を最大3並列で実行し、全列一致を評価の必須条件にする。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import shutil
from typing import Any

from scripts.collect_e34c import OUT
from scripts.run_e17_ablation_20260928 import ALL_SOURCES
from scripts.run_e3_exchange_eval_20260926 import save_json

WORKERS = 3


def launch(source: str) -> None:
    """完成済み入力を保持し、別記録の停止通知後は新しい処理を始めない。"""
    if (OUT/'STOP.json').exists():
        return
    if (OUT/'checks'/f'{source}.full.json').exists():
        return
    with (OUT/f'collect_{source}.log').open('a') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'scripts.collect_e34c', '--source', source],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def collect_all() -> None:
    """全5記録が一致して初めて後続の比較を許可する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT/'PREREGISTRATION.md').exists()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(launch, ALL_SOURCES))
    checks = {s: json.loads((OUT/'checks'/f'{s}.full.json').read_text()) for s in ALL_SOURCES}
    save_json(OUT/'INPUT_CHECK.json', dict(sources=checks,
        rows=sum(v['rows'] for v in checks.values()),
        matching_rows=sum(v['matching_rows'] for v in checks.values()),
        passed=all(v['passed'] for v in checks.values())))


def replay_one(task: tuple[str, str]) -> None:
    """最大3並列の各プロセスへ同じ原票を渡す。"""
    source, variant = task
    with (OUT/f'{variant}_{source}.log').open('a') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'scripts.run_e34c',
            '--phase', variant, '--source', source],
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def audited_worker(source: str, variant: str) -> None:
    """キャッシュ欠測時も再生器が実際に再計算したDを旧値と照合する。"""
    import numpy as np
    from scripts import run_e34b
    from src.exchange_event_record import read_records, static_key
    old = {r['key']: r['input']['d_features']
           for r in read_records(Path('logs/e31/records')/f'{source}.jsonl.gz') if r['kind'] == 'static'}
    new = {r['key'] for r in read_records(OUT/'records'/f'{source}.jsonl.gz') if r['kind'] == 'static'}
    original = run_e34b.replay_module.static_builder
    checked: set[str] = set()
    def builder(record: Path) -> Any:
        build = original(record)
        def audited(boards: tuple, snapshot: Any, elapsed: float, m0: float) -> Any:
            value = build(boards, snapshot, elapsed, m0)
            key = static_key(boards, snapshot)
            if key in old:
                np.testing.assert_array_equal(value.d_features, old[key], err_msg=key)
                checked.add(key)
            return value
        return audited
    run_e34b.OUT = OUT
    run_e34b.replay_module.static_builder = builder
    run_e34b.worker(source, variant)
    missing = old.keys()-new
    save_json(OUT/'checks'/f'{source}.{variant}.static.json', dict(
        checked=len(checked), recomputed=len(checked & missing),
        unused_old_only=len(missing-checked), old_only=len(missing), passed=True))


def evaluate() -> None:
    """旧E32の採点行を固定したまま、起点ガードだけを対比較する。"""
    inputs = json.loads((OUT/'INPUT_CHECK.json').read_text())
    assert inputs['passed'] and 'record_kinds' in inputs and 'feature_cache' in inputs
    (OUT/'cohort').mkdir(exist_ok=True)
    for path in Path('logs/e33b/cohort').glob('*.json'):
        shutil.copyfile(path, OUT/'cohort'/path.name)
    shutil.copyfile('logs/e33b/COHORT.json', OUT/'COHORT.json')
    for variant in ('off', 'on'):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(replay_one, [(s, variant) for s in ALL_SOURCES]))
        if variant == 'off':
            from scripts.report_e34c import old_equivalence
            save_json(OUT/'OLD_EQUIVALENCE.json', old_equivalence())
    from scripts.report_e34c import report
    print(report(), flush=True)


def main() -> None:
    """収集の全行検収と評価実行を明示的に分離する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('collect', 'evaluate', 'off', 'on'), default='collect')
    parser.add_argument('--source', choices=ALL_SOURCES)
    args = parser.parse_args()
    if args.phase in ('off', 'on'):
        assert args.source and json.loads((OUT/'INPUT_CHECK.json').read_text())['passed']
        audited_worker(args.source, args.phase)
    elif args.phase == 'collect':
        launch(args.source) if args.source else collect_all()
    else:
        evaluate()


if __name__ == '__main__':
    main()
