"""署名を保持した収集を旧原票と逐次照合し、300行で停止判定する。"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from scripts import collect_e34b as capture
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.enrich_e31_snapshots import fire_keys
from scripts.e34c_observations import Observations
from src.exchange_event_record import ExchangeEventRecorder, encode

OUT = Path('logs/e34c')
PREFIX_ROWS = 300
CORE_FIELDS = ('confirmed_board', 'state', 'score', 'next_pair', 'dnext_pair')
OUTPUT_FLAGS = ('--out', '--dump-timeline', '--dump-display-timeline',
                '--dump-exchange-events', '--exchange-event-record', '--review-data-csv')


def command(source: str) -> list[str]:
    """評価モデル・評価スイッチも実際の旧収集コマンドから維持する。"""
    status = capture.original_status(source) if source != 'review' else (
        capture.ROOT/'logs/review_zenchi_g41_43_e14/status.json')
    args = json.loads(status.read_text())['command'][3:]
    args = [value for value in args if value not in ('--worker', '--review-data-panel')]
    for flag in OUTPUT_FLAGS:
        if flag in args:
            index = args.index(flag)+1
            args[index] = str(OUT.resolve()/'capture'/source/Path(args[index]).name)
    if '--no-render' not in args:
        args.append('--no-render')
    return args


def updates(source: str) -> Iterator[dict]:
    """補完済み旧入力の認識行を生の列構造で読む。"""
    with gzip.open(Path('logs/e31/records')/f'{source}.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'update':
                yield row


def differences(left: Any, right: Any, path: str = '') -> list[str]:
    """保持印のみ除外し、欠けた列も不一致として扱う。"""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            if key == 'prefire_origin_hold':
                continue
            name = f'{path}.{key}'
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [p for i, (a, b) in enumerate(zip(left, right))
                for p in differences(a, b, f'{path}[{i}]')]
    return [] if left == right else [path]


class InputCheck:
    """新原票の各行を旧入力へ対応付け、最初の300行を同期的に検収する。"""

    def __init__(self, source: str, checkpoint: bool = True) -> None:
        self.source, self.old = source, updates(source)
        self.checkpoint = checkpoint
        self.rows, self.matches, self.core_matches = 0, 0, 0
        self.columns: Counter[str] = Counter()
        self.first: dict[str, Any] = {}

    def observe(self, row: dict) -> None:
        """原票を書いた直後に照合し、全収集プロセスへ停止を通知する。"""
        if (OUT/'STOP.json').exists():
            raise RuntimeError('別記録の300行照合が不一致のため停止')
        if row['kind'] != 'update':
            return
        old, new = next(self.old, None), encode(row)
        changed = differences(old, new)
        core = old is not None and old['args']['tuple'][3] == new['args']['tuple'][3]
        if core:
            for side in ('p1', 'p2'):
                a, b = (r['args']['tuple'][0]['namespace'][side]['namespace'] for r in (old, new))
                core = core and all(a.get(k) == b.get(k) for k in CORE_FIELDS)
        self.rows += 1
        self.matches += not changed
        self.core_matches += core
        self.columns.update(changed)
        if changed and not self.first:
            self.first = dict(row=self.rows, columns=changed, old=old, new=new)
        if self.rows == PREFIX_ROWS and self.checkpoint:
            result = self.summary()
            save_json(OUT/'checks'/f'{self.source}.prefix.json', result)
            if self.core_matches != PREFIX_ROWS:
                save_json(OUT/'STOP.json', result)
                raise RuntimeError('最初の300行の確定盤面・状態・得点・NEXTが不一致')

    def summary(self) -> dict:
        """一致数と不一致列を母数ごと保存する。"""
        return dict(source=self.source, rows=self.rows, matching_rows=self.matches,
                    core_matching_rows=self.core_matches, differences=self.columns, first=self.first)

    def finish(self) -> None:
        """末尾の余剰・欠測も検出して全区間の検収結果を保存する。"""
        extra = sum(1 for _ in self.old)
        result = dict(self.summary(), old_extra_rows=extra)
        result['passed'] = self.rows == self.matches and extra == 0
        save_json(OUT/'checks'/f'{self.source}.full.json', result)


def has_before_board(source: str) -> bool:
    """E8/E10cとE14の記録契約の違いを、旧原票の最初の通知で確認する。"""
    stream = updates(source)
    try:
        for row in stream:
            for side in ('p1', 'p2'):
                event = row['args']['tuple'][0]['namespace'][side]['namespace']['chain_event']
                if event is not None:
                    return 'before_board' in event['namespace']
    finally:
        stream.close()
    raise ValueError(f'連鎖通知がなく保存契約を確認できない: {source}')


def record_schema(row: dict, required: set[tuple], keep_before_board: bool = False) -> None:
    """E31原票と同じ保存契約を使い、E34で追加された入力を混入させない。"""
    if row['kind'] != 'update':
        return
    result, game = row['args'][0], row['args'][4]
    for idx, side in enumerate((result.p1, result.p2)):
        # E26は観測が無い列もNoneを明記していた。
        if not hasattr(side, 'midchain_board'):
            side.midchain_board = None
        event = side.chain_event
        # E8/E10cには列がなく、E14には存在する。各原票の起点選択を維持する。
        if not keep_before_board and event is not None and hasattr(event, 'before_board'):
            del event.before_board
        # 画像窓の対象もE31と同じE27発火一覧から決める。
        key = (game, idx, event.trigger_sec) if event is not None else None
        if hasattr(side, 'prefire_snapshot') and key not in required:
            del side.prefire_snapshot


def collect(source: str) -> None:
    """既存の収集と試合範囲・画像窓取得を変更せず、原票だけを監査する。"""
    capture.OUT = OUT.resolve()
    capture.command = command
    capture.SnapshotCapture = lambda video, dest: Observations(video, dest, source)
    check, original = InputCheck(source), ExchangeEventRecorder.write
    required = fire_keys(source)
    keep_before_board = has_before_board(source)
    def write(recorder: Any, row: dict) -> None:
        record_schema(row, required, keep_before_board)
        original(recorder, row)
        check.observe(row)
    ExchangeEventRecorder.write = write
    capture.collect(source)
    check.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--schema-only', action='store_true')
    options = parser.parse_args()
    if options.schema_only:
        print(dict(source=options.source, before_board=has_before_board(options.source)))
    else:
        collect(options.source)
