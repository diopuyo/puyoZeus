"""E32とE34の対比較用に、保持印を含む新しい認識入力を収集する。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts.run_e3_exchange_eval_20260926 import ROOT, SOURCES, save_json, digest
from scripts.e34b_snapshot_capture import SnapshotCapture
from src.exchange_event_record import ExchangeEventRecorder
from src.match_range_gate import build
from src.recognition_pipeline import RecognitionPipeline

OUT = ROOT/'logs/e34b'
CLI_FLAGS = ('--exchange-event-live-count', '--exchange-event-death-guard',
    '--confirmed-death-hold', '--death-formula-guard', '--color-score-safety',
    '--multi-landing-death', '--death-pending-ledger', '--hidden-row-death',
    '--midchain-completion', '--prefire-snapshot', '--hidden-row-belief')


def original_status(source: str) -> Path:
    """動画・区間・ウォームアップは元の保存コマンドから取得する。"""
    if source in SOURCES:
        return ROOT/'logs/e8/renders'/source/'on/status.json'
    return ROOT/('logs/review_zenchi_g41_43_e27/status.json' if source == 'review'
        else 'logs/review_zenchi_part3/on_e10c/status.json')


def command(source: str) -> list[str]:
    """認識関連引数を維持し、出力先とE32評価条件だけを指定する。"""
    status = original_status(source)
    args = json.loads(status.read_text())['command'][3:]
    previous, target = str(status.parent), str(OUT/'capture'/source)
    args = [v.replace(previous, target) for v in args if v != '--worker']
    if '--review-data-panel' in args:
        args.remove('--review-data-panel')
    if '--review-data-csv' in args:
        index = args.index('--review-data-csv')
        del args[index:index+2]
    for flag in CLI_FLAGS:
        if flag not in args:
            args.append(flag)
    if '--exchange-event-model-dir' not in args:
        args.extend(['--exchange-event-model-dir', 'models/exchange_event_v3'])
    else:
        args[args.index('--exchange-event-model-dir')+1] = 'models/exchange_event_v3'
    if '--no-render' not in args:
        args.append('--no-render')
    return args


def boundaries(video: Path) -> Any:
    """収集側の既存生成手段を既定設定のまま使う。"""
    gate = build(video.stem, ROOT, True)
    if gate.active:
        return gate
    target = ROOT/'data/verify/match_boundaries_v5'/video.stem/'matches.tsv'
    if not target.exists():
        subprocess.run([sys.executable, '-m', 'scripts.count_match_v4', '--video', str(video),
            '--out-root', str(ROOT/'data/verify/match_boundaries_v5')], check=True)
    gate = build(video.stem, ROOT, True)
    if not gate.active:
        # W4の既存代替手段。v4が機能しない動画だけを既定条件で再走する。
        subprocess.run([sys.executable, '-m', 'scripts.count_match_via_ocr', '--video', str(video),
            '--out', str(target)], check=True)
        gate = build(video.stem, ROOT, True)
    if not gate.active:
        raise ValueError(f'既存v4/OCR境界検出で有効な試合範囲なし: {video}')
    return gate


def install_capture(gate: Any) -> None:
    """保持印は認識時点で記録し、試合範囲をヘッダーに固定保存する。"""
    load, update, write = RecognitionPipeline.load_default, ExchangeEventRecorder.update, ExchangeEventRecorder.write
    def configured(**kwargs: Any) -> RecognitionPipeline:
        return load(**dict(kwargs, enable_landing_chain_record_hold=True,
                           enable_chain_active_record_hold=True))
    def captured(self: Any, result: Any, *args: Any) -> None:
        marked = replace(result, p1=replace(result.p1, prefire_origin_hold=result.p1.landing_chain_started),
                         p2=replace(result.p2, prefire_origin_hold=result.p2.landing_chain_started))
        update(self, marked, *args)
    def recorded(self: Any, row: dict) -> None:
        if row['kind'] == 'header':
            row = dict(row, prefire_match_ranges=dict(ranges=gate.ranges, source=gate.source,
                sha256=digest(Path(gate.source))))
        write(self, row)
    RecognitionPipeline.load_default = configured
    ExchangeEventRecorder.update, ExchangeEventRecorder.write = captured, recorded


def collect(source: str) -> None:
    """独立プロセスで一記録を収集し、完了した原票だけを公開する。"""
    import fcntl
    from scripts.run_e3_exchange_eval_20260926 import worker
    args = command(source)
    video = Path(args[args.index('--video')+1])
    # reviewとzenchiの共通境界ファイルは一度だけ生成する。
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/f'{video.stem}.boundary.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        gate = boundaries(video)
    install_capture(gate)
    snapshots = SnapshotCapture(video, OUT/'records'/f'{source}.windows.json.gz')
    snapshots.install()
    dest = OUT/'records'/f'{source}.jsonl.gz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    args[args.index('--exchange-event-record')+1] = str(dest.with_suffix('.tmp'))
    save_json(OUT/'capture'/source/'COMMAND.json', dict(args=args, match_source=gate.source))
    sys.argv = ['collect_e34b', '--worker', *args]
    worker()
    snapshots.close()
    dest.with_suffix('.tmp').replace(dest)
    save_json(dest.with_suffix('.json'), dict(state='completed', source=source,
        record_sha256=digest(dest), match_source=gate.source, command=args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=(*SOURCES, 'review', 'zenchi'), required=True)
    collect(parser.parse_args().source)
