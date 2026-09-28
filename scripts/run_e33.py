"""E33の事前登録済みON/OFF・初回誤差・打ち切り再生を実行する。"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from scripts import run_e32 as e32
from scripts.audit_e32b import serial
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e3_exchange_eval_20260926 import save_json, SOURCES

OUT = Path('logs/e33')
RECORDS = Path('logs/e31/records')
CUTOFF = 816.0
CUTOFF_SOURCE = 'fcXG83vInDY'
WORKERS = 2
OPTIONS = dict(e32.OPTIONS, prefire_stage_timeout=True)


class TimeoutTrace(e32.BeliefTrace):
    """初回得点を保存したまま、最後の有効予測と因果状態を別記する。"""
    def __init__(self, source: str, out: Path, event_hashes: bool = True) -> None:
        super().__init__(source)
        self.out = out
        self.event_hashes = event_hashes
        self.hashes: list[dict] = []

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        for entry in self.engine.entries.values():
            chain, row = entry['chain'], entry['audit']
            if row['accepted'] and self.engine.active(chain) is not None:
                row['last_prediction'] = entry['stats']['mean_score']
                row['last_prediction_sec'] = inputs[3]
            elif row['withdrawn'] and chain.end_signal_sec is None:
                row['last_prediction'] = chain.predicted_final_score
                row['last_prediction_sec'] = inputs[3]
        if self.event_hashes and self.source == CUTOFF_SOURCE and inputs[3] <= CUTOFF:
            payload = json.dumps(dict(records=overlay.tracker.records,
                diagnostics=overlay.tracker.diagnostics), default=serial,
                ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            self.hashes.append(dict(t=inputs[3], hash=hashlib.sha256(payload.encode()).hexdigest()))
            if inputs[3] == CUTOFF:
                overlay.tracker.save(self.out/'checkpoint.jsonl')

    def summary(self) -> dict:
        """最後の予測誤差は参考指標として初回誤差と分離する。"""
        value = super().summary()
        for row in value['rows']:
            predicted = row.get('last_prediction', row['predicted_score'])
            row['last_error'] = (predicted-row['final_score']
                if predicted is not None and row['final_score'] is not None else None)
        save_json(self.out/'state_hashes.json', self.hashes)
        return value


def worker(source: str, control: bool, cutoff: bool = False) -> None:
    """基準と同じ入力・モデルを独立したプロセスで再生する。"""
    prior = e32.e31.prior
    prior.OUT = OUT
    variant = 'prefix' if cutoff else 'off' if control else 'on'
    dest = prior.directory(variant, source)
    prior.AuditTrace = lambda name: TimeoutTrace(name, dest, not control)
    def enriched(record: Path, target: Path, *args: Any, **kwargs: Any) -> dict:
        """初回採用・最終予測・イベント状態を同じ再生から採取する。"""
        path = OUT/'input_prefix.jsonl.gz' if cutoff else RECORDS/record.name
        value = replay(path, target, *args, **kwargs)
        save_json(target/'snapshot_final_audit.json', args[2].summary())
        return value
    prior.replay = enriched
    prior.locked_worker(variant, source, e32.OPTIONS if control else OPTIONS)
    if control:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        save_json(OUT/f'OFF_{source}.json', compare(Path('logs/e32/on')/suffix, dest))


def truncate() -> None:
    """未来の静止特徴も除く、入力の連続接頭辞を作る。"""
    frames = 0
    with gzip.open(RECORDS/f'{CUTOFF_SOURCE}.jsonl.gz', 'rt') as source:
        with gzip.open(OUT/'input_prefix.jsonl.gz', 'wt') as target:
            for line in source:
                row = json.loads(line)
                if row['kind'] == 'complete':
                    break
                if row['kind'] == 'update':
                    if row['args']['tuple'][3] > CUTOFF:
                        break
                    frames += 1
                target.write(line)
            target.write(json.dumps(dict(kind='complete', frames=frames))+'\n')


def launch(task: tuple) -> None:
    """既存D1診断との共存のため評価子プロセスは最大二つに制限する。"""
    source, control, cutoff = task
    name = f'{source}_{control}_{cutoff}'
    command = [sys.executable, '-m', 'scripts.run_e33', '--source', source]
    command += ['--control'] if control else []
    command += ['--cutoff'] if cutoff else []
    with (OUT/f'{name}.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """再生完了後に固定閾値でのみ採点する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=e32.e31.prior.ALL_SOURCES)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--cutoff', action='store_true')
    args = parser.parse_args()
    assert (OUT/'PREREGISTRATION.md').exists()
    if args.source:
        worker(args.source, args.control, args.cutoff)
        return
    truncate()
    tasks = [(s, c, False) for c in (False, True) for s in (CUTOFF_SOURCE, *SOURCES[:1], 'review', 'zenchi', SOURCES[2])]
    tasks.append((CUTOFF_SOURCE, False, True))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(launch, tasks))
    from scripts.report_e33 import report
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
