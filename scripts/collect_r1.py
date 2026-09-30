"""E34cの署名・原票契約を維持し、R1一フラグだけを変えて再収集する。"""
from __future__ import annotations
import argparse
from functools import wraps
import json
from pathlib import Path
import sys
from typing import Any
from scripts import collect_e34b as capture, collect_e34c as checked
from scripts.e34c_observations import Observations
from scripts.enrich_e31_snapshots import fire_keys
from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from src.exchange_event_record import ExchangeEventRecorder
from src.recognition_pipeline import RecognitionPipeline

OUT = Path('logs/r1')


class R1Observations(Observations):
    """新しい盤面で発火条件を満たした連鎖も、過去画像窓を欠測にしない。"""

    def observed(self, result: Any, stamp: float, game: int) -> Any:
        """既存の因果的画像窓を全開始通知へ付け、旧発火一覧で切り落とさない。"""
        for side, value in enumerate((result.p1, result.p2)):
            if value.chain_event is not None:
                self.required.add((game, side, value.chain_event.trigger_sec))
        return super().observed(result, stamp, game)


def install_capture(gate: Any) -> None:
    """E34の保持フラグを追加せず、署名を維持したまま全実効設定を保存する。"""
    load, write = RecognitionPipeline.load_default, ExchangeEventRecorder.write
    @wraps(load)
    def configured(**kwargs: Any) -> RecognitionPipeline:
        dest = Path(sys.argv[sys.argv.index('--out')+1]).parent
        save_json(dest/'effective_config.json', kwargs)
        return load(**kwargs)
    def recorded(recorder: Any, row: dict) -> None:
        if row['kind'] == 'header':
            row = dict(row, prefire_match_ranges=dict(ranges=gate.ranges, source=gate.source,
                                                    sha256=digest(Path(gate.source))))
        write(recorder, row)
    RecognitionPipeline.load_default, ExchangeEventRecorder.write = configured, recorded


def collect(source: str, mode: str) -> None:
    """ONは新観測のみ、OFFは全認識行を元のE31原票へ同期照合する。"""
    root = OUT if mode == 'on' else OUT/'off_capture'
    audit = None
    if mode == 'on':
        from scripts.r1_correction_audit import CorrectionAudit
        audit = CorrectionAudit(root/'followup'/f'{source}.jsonl')
        audit.install()
    checked.OUT = root
    capture.OUT = root.resolve()
    original_command = checked.command
    def command(name: str) -> list[str]:
        args = original_command(name)
        return args + (['--placement-signal-reconcile'] if mode == 'on' else [])
    capture.command, capture.install_capture = command, install_capture
    required, before = fire_keys(source), checked.has_before_board(source)
    def snapshots(video: Path, dest: Path) -> Observations:
        reader = (R1Observations if mode == 'on' else Observations)(video, dest, source)
        reader.required = required
        return reader
    capture.SnapshotCapture = snapshots
    check = checked.InputCheck(source) if mode == 'off' else None
    original = ExchangeEventRecorder.write
    def write(recorder: Any, row: dict) -> None:
        checked.record_schema(row, required, before)
        original(recorder, row)
        if check is not None:
            check.observe(row)
    ExchangeEventRecorder.write = write
    capture.collect(source)
    if audit is not None:
        audit.close()
    if check is not None:
        check.finish()
    save_json(root/'completed'/f'{source}.json', dict(source=source, mode=mode, completed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--mode', choices=('off', 'on'), required=True)
    args = parser.parse_args()
    assert json.loads((OUT/'stage2/SUMMARY.json').read_text())['passed']
    collect(args.source, args.mode)
