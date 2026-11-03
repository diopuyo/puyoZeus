"""zenchi s1の全更新を保存・ビット照合する。勝敗の採点はしない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import struct

import numpy as np
import pandas as pd

from scripts.b3_time_gate import MODEL, ROOT, TABLE, Replay
from src.b3_source import iter_frames, resolve_attempt, sha256_file
from src.b3_timing import TimingTable
from src.b3_train_model import load_model

PROGRESS_ROWS = 1000


def assert_bits(before: dict, after: dict) -> None:
    """浮動小数点はNaN位置に加え符号・ペイロードも含めて比較する。"""
    if before.keys() != after.keys():
        raise AssertionError('列が不一致')
    for name, value in before.items():
        actual = after[name]
        if isinstance(value, (float, np.floating)):
            equal = struct.pack('d', value) == struct.pack('d', actual)
        else:
            equal = value == actual
        if not equal:
            raise AssertionError((before.get('row_key'), name, value, actual))


def reference_predictions(path: Path | None, record: Path) -> dict[int, float]:
    """同じモデル・入力・時刻表の変更前原票を再利用できる場合だけ読む。"""
    if path is None:
        return {}
    metadata = json.loads((path.parent / 'metadata.json').read_text())
    assert metadata['model_sha256'] == sha256_file(MODEL)
    assert metadata['timing_sha256'] == sha256_file(TABLE)
    for name, digest in metadata['input_sha256'].items():
        assert digest == sha256_file(record / name)
    return {r['frame']: r['p'] for line in path.read_text().splitlines() if (r := json.loads(line))}


def run(mode: str, path: Path, publications: Path | None = None) -> dict:
    """変更前の独立ソースで保存し、変更後は同じ全フレームを再生して比較する。"""
    video = 'b3_zenchi_s1'
    record = resolve_attempt(video, ROOT / 'b3_eval_records').record
    replay = Replay(record, video, TimingTable.from_json(TABLE.read_text()))
    model = load_model(MODEL)
    saved = reference_predictions(publications, record)
    count, games, columns = 0, set(), 0
    with path.open('xb' if mode == 'save' else 'rb') as stream:
        for frame in iter_frames(record):
            row, _ = replay.step(frame)
            if row is None:
                continue
            if mode == 'save':
                p = (saved.pop(row['frame_idx']) if publications is not None
                     else float(model.predict(pd.DataFrame([row]))[0]))
                pickle.dump((row, p), stream, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                expected, old_p = pickle.load(stream)
                assert_bits(expected, row)
                p = float(model.predict_row(row))
                assert_bits({'p': old_p}, {'p': p})
            count += 1
            columns = len(row)
            games.add(row['game_idx'])
            if count % PROGRESS_ROWS == 0:
                print(json.dumps(dict(mode=mode, rows=count, frame=frame['frame'])), flush=True)
        if mode == 'compare' and stream.read(1):
            raise AssertionError('変更後の更新数が不足')
    if saved:
        raise AssertionError('変更前の公開原票に未照合の更新がある')
    return dict(mode=mode, rows=count, games=len(games), columns=columns,
                publications=str(publications) if publications else None,
                model_sha256=sha256_file(MODEL), timing_sha256=sha256_file(TABLE),
                all_columns_bit_equal=mode == 'compare', prediction_bit_equal=mode == 'compare')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('save', 'compare'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--publications', type=Path)
    args = parser.parse_args()
    result = run(args.mode, args.path, args.publications)
    args.path.with_suffix('.'+args.mode+'.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
