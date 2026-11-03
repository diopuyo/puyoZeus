"""review短区間を描画CLI構築済みoverlayで再生し、段2b保存表示と照合する。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import numpy as np

OUT = Path('logs/static_gfe_cli_20261002')
SAVED = Path('logs/gfe_phase_stage2b_20261002')
CANDIDATE = Path('models/exchange_event_v1_r1_hgb')
DURATION_SEC = 90.
MIN_AVAILABLE = 6 * 1024**3


def memory_available() -> int:
    """6GiB以上の余裕を確認する。"""
    lines = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(lines['MemAvailable'].split()[0]) * 1024
    if available < MIN_AVAILABLE:
        raise RuntimeError(f'メモリ不足: {available}')
    return available


def short_record() -> tuple[Path, dict]:
    """先頭から状態を再現し、90秒区間の完了マーカーを付ける。"""
    from scripts.gfe_phase_stage2_replay_20261002 import record_path
    from src.exchange_event_record import read_records, encode
    path = OUT/'review.jsonl.gz'
    stream = read_records(record_path('review'))
    header, start, frames = next(stream), None, 0
    with gzip.open(path, 'wt', encoding='utf-8') as target:
        target.write(json.dumps(encode(header))+'\n')
        for row in stream:
            if row['kind'] == 'complete':
                break
            if row['kind'] == 'update':
                stamp = row['args'][3]
                start = stamp if start is None else start
                if stamp > start + DURATION_SEC:
                    break
                frames += 1
            target.write(json.dumps(encode(row))+'\n')
        target.write(json.dumps(dict(kind='complete', frames=frames))+'\n')
    return path, header


class Ready(Exception):
    """動画I/O直前に、実際に構築されたoverlayを取り出す。"""


def render_overlay(flags: list[str], record: Path, header: dict) -> Any:
    """main→argparse→generate→overlay→trackerを実行する。"""
    from scripts import visualize_advantage_overlay as render
    from scripts.replay_exchange_event_20260926 import static_builder
    builder, original, result = static_builder(record), render.ExchangeEventOverlay, []

    def capture(*args: Any, **kwargs: Any) -> Any:
        result.append(original(*args, **kwargs))
        raise Ready

    extras = ['--per-side-settled'] if header['per_side_settled'] else []
    # 死亡信号は保存記録から供給するため、画像用テンプレートは読み込まない。
    with patch.object(sys, 'argv', ['render', *flags, *extras]), \
            patch('src.exchange_event_terminal.ObservedDeathDetector', return_value=None), \
            patch.object(render, '_exchange_static_input', builder), \
            patch.object(render, 'ExchangeEventOverlay', capture):
        try:
            render.main()
        except Ready:
            return result[0]
    raise AssertionError('overlay構築経路を通らなかった')


def saved_display(mode: str, destination: Path) -> dict:
    """保存済み段2bの全表示列を同じ時刻まで切り、dtypeとバイトを比較する。"""
    with np.load(destination/'display.npz') as actual, np.load(
            SAVED/mode/'replay/review/display.npz') as expected:
        assert actual.files == expected.files
        count = len(actual['t_sec'])
        for name in actual.files:
            left, right = actual[name], expected[name]
            if right.ndim and len(right) == len(expected['t_sec']):
                right = right[:count]
            assert left.dtype == right.dtype and left.shape == right.shape, name
            assert left.tobytes() == right.tobytes(), name
        return dict(rows=count, start=float(actual['t_sec'][0]), end=float(actual['t_sec'][-1]),
                    all_columns_byte_identical=True)


def run(mode: str, record: Path, header: dict) -> dict:
    """CLI引数による配線を使い、推論器を差し替えずに用途を監査する。"""
    from scripts import replay_exchange_event_20260926 as replay
    from src.exchange_event_evaluator import FileExchangeModels
    from src import production_config
    import shlex
    flags = shlex.split(production_config.exchange_event_flags())
    if mode == 'R1s':
        flags += ['--exchange-event-static-gfe-dir', str(CANDIDATE)]
    overlay = render_overlay(flags, record, header)
    counts: Counter = Counter()
    original = FileExchangeModels.predict_source_probability

    def predict(models: FileExchangeModels, name: str, values: np.ndarray) -> float:
        purpose = 'landing' if models is overlay.tracker.models.landing_models else 'static_or_s3'
        counts[(purpose, name, type(models.estimators[name]).__name__)] += 1
        return original(models, name, values)

    destination = OUT/mode
    destination.mkdir(exist_ok=True)
    with patch.object(sys, 'argv', ['replay', str(record), '--out', str(destination), *flags]), \
            patch.object(replay, 'ExchangeEventOverlay', lambda *a, **kw: overlay), \
            patch.object(FileExchangeModels, 'predict_source_probability', predict):
        replay.main()
    result = saved_display(mode, destination)
    reference = OUT/(mode+'_native_replay')
    reference.mkdir(exist_ok=True)
    with patch.object(sys, 'argv', ['replay', str(record), '--out', str(reference), *flags]):
        replay.main()
    result['native_replay_identity'] = replay.compare(reference, destination)
    result['calls'] = [{'purpose': p, 'model': n, 'estimator': k, 'count': v}
                       for (p, n, k), v in counts.items()]
    result['available_bytes'] = memory_available()
    if mode == 'R1s':
        assert counts[('landing', 'G_fe', 'LogisticRegression')] > 0
        assert counts[('static_or_s3', 'G_fe', 'HistGradientBoostingClassifier')] > 0
    return result


def main() -> None:
    """本番引数をそのまま使い、資産と実装は変更しない。"""
    os.nice(max(0, 10-os.nice(0)))
    available = memory_available()
    OUT.mkdir(parents=True, exist_ok=True)
    record, header = short_record()
    result = dict(nice=os.nice(0), available_bytes=available, duration_sec=DURATION_SEC)
    result['modes'] = {mode: run(mode, record, header) for mode in ('baseline', 'R1s')}
    (OUT/'RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
