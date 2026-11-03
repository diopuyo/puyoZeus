"""全対象フレームへ確定盤面・NEXT・会計原票を結合する。"""
from __future__ import annotations

import csv
from bisect import bisect_right
import json
from pathlib import Path
from typing import Any

import cv2
from scripts._diag_review1003_extract import ROOT, OUT, WINDOWS, resources
from scripts._diag_review1003_inspect import TIMES
from src.exchange_event_record import read_records, encode

FRAME_RATE = 30


def compact(board: Any) -> str:
    """隠し段から最下段の13行をスラッシュで区切る。"""
    return '/'.join(''.join('?' if int(c) == 10 else str(int(c)) for c in row)
                    for row in board._grid)


def extract(game: int, start: float, end: float) -> None:
    """時刻の未来参照をせず、最後のSTABLE入力を各出力へ添付する。"""
    work = ROOT / 'work' / f'set1_game{game:02}'
    latest, updates = [None, None], {}
    for item in read_records(work / 'inputs.jsonl.gz'):
        if item['kind'] != 'update':
            continue
        args = item['args']
        result, snapshot, finalization, stamp = args[:4]
        for idx, side in enumerate((result.p1, result.p2)):
            if side.state.name == 'STABLE' and side.confirmed_board is not None:
                latest[idx] = dict(t_sec=stamp, board=compact(side.confirmed_board),
                                  next=side.next_pair, dnext=side.dnext_pair)
        if stamp <= end:
            if stamp < start:
                updates.clear()
            updates[stamp] = dict(t_sec=stamp, boards=latest.copy(),
                snapshot=vars(snapshot), finalization=vars(finalization), raw=encode(args))
    rows = list(csv.DictReader((OUT / f'game{game:02}_frames.csv').open()))
    stamps = sorted(updates)
    for row in rows:
        stamp = float(row['t_sec'])
        value = updates[stamps[bisect_right(stamps, stamp) - 1]]
        row['inputs'] = value
    (OUT / f'game{game:02}_joined.json').write_text(json.dumps(rows), encoding='utf-8')
    samples = [min(rows, key=lambda r: abs(float(r['t_sec'])-t)) for t in TIMES[game]]
    (OUT / f'game{game:02}_samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    capture = cv2.VideoCapture(str(ROOT / 'review_part01.mp4'))
    for row in samples:
        stamp = float(row['t_sec'])
        capture.set(cv2.CAP_PROP_POS_MSEC, float(row['review_sec']) * 1000)
        ok, frame = capture.read()
        assert ok, stamp
        cv2.imwrite(str(OUT / f'game{game:02}_{stamp:.3f}.jpg'), frame)
    capture.release()
    print(game, len(rows), len(updates), flush=True)


def main() -> None:
    """3区間を逐次処理する。"""
    for window in WINDOWS:
        resources()
        extract(*window)


if __name__ == '__main__':
    main()
