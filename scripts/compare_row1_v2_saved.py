"""verify_row1_actionlog.py の保存実画素再生を比較する。信頼済みpickleだけを入力する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.board import BOARD_COLS, BOARD_ROWS

SIDES = ('p1', 'p2')
TARGET = (1, 2)
LANDING_FRAME = 31734
LANDING_COLOR = 2
LANDING_SIDE = 1
LANDING_END = 31755
SETS = (('video_65_1054.0', 46), ('video_65_2327.0', 41), ('video_79_1758.0', 41))
PROBABILITY_INDEX = 1
PERCENT = 100
EVEN_PROBABILITY = .5


def read(path: Path) -> list[dict[str, Any]]:
    """既存の再生台の出力を読む。"""
    return pickle.loads((path / 'rows.pickle').read_bytes())


def board_diffs(left: Any, right: Any) -> list[dict[str, Any]]:
    """盤面なしもセル差として数え、位置を全件残す。"""
    a = np.full((BOARD_ROWS, BOARD_COLS), -1) if left is None else left._grid
    b = np.full((BOARD_ROWS, BOARD_COLS), -1) if right is None else right._grid
    return [dict(row=int(r), col=int(c), off=int(a[r, c]), on=int(b[r, c]))
            for r, c in np.argwhere(a != b)]


def compare(off: list[dict], on: list[dict]) -> dict[str, Any]:
    """公開盤以外の経路と、表示履歴が分かれたコマも照合する。"""
    assert len(off) == len(on) > 0
    differences, display = [], []
    counts = dict(internal_frames=0, inferred_sides=0, cnn_sides=0, states=0)
    for a, b in zip(off, on):
        assert a['frame'] == b['frame']
        left, right = pickle.loads(a['result']), pickle.loads(b['result'])
        counts['internal_frames'] += a['internal'] != b['internal']
        for side, name in enumerate(SIDES):
            sa, sb = getattr(left, name), getattr(right, name)
            cells = board_diffs(sa.confirmed_board, sb.confirmed_board)
            if cells:
                differences.append(dict(frame=a['frame'], side=side + 1, cells=cells))
            counts['inferred_sides'] += bool(board_diffs(sa.inferred_board, sb.inferred_board))
            counts['cnn_sides'] += bool(board_diffs(sa.cnn_board, sb.cnn_board))
            counts['states'] += sa.state != sb.state
        pa, pb = (float(pickle.loads(row['display'])[PROBABILITY_INDEX]) for row in (a, b))
        display.append(dict(frame=a['frame'], off=pa, on=pb, pp=abs(pa-pb)*PERCENT))
    deltas = [d['pp'] for d in display]
    return dict(frames=len(off), compared_cells=len(off)*len(SIDES)*BOARD_ROWS*BOARD_COLS, **counts,
        raw_prediction_different_frames=sum(a['raw'] != b['raw'] for a, b in zip(off, on)),
        target_cells=sum((c['row'], c['col']) == TARGET for d in differences for c in d['cells']),
        outside_cells=sum((c['row'], c['col']) != TARGET for d in differences for c in d['cells']),
        differences=differences, display_different_frames=sum(d > 0 for d in deltas),
        display_max_pp=max(deltas), display_p95_pp=float(np.percentile(deltas, 95)),
        display_sign_flips=sum((d['off']-EVEN_PROBABILITY)*(d['on']-EVEN_PROBABILITY) < 0 for d in display),
        display_differences=[d for d in display if d['pp'] > 0])


def first_landing(rows: list[dict]) -> dict[str, Any]:
    """元動画のコマ番号で遅れを数える。10Hzのサンプル番号では数えない。"""
    for row in rows:
        if not LANDING_FRAME <= row['frame'] <= LANDING_END:
            continue
        result = pickle.loads(row['result'])
        board = getattr(result, SIDES[LANDING_SIDE]).confirmed_board
        if board is not None and board.get(*TARGET) == LANDING_COLOR:
            return dict(first_public_frame=row['frame'], delay_source_frames=row['frame']-LANDING_FRAME)
    return dict(first_public_frame=None, delay_source_frames=None)


def positive_counts(rows: list[dict]) -> list[int]:
    """冷起動で幻が再現しなかった母数も隠さない。"""
    counts = [0 for _ in SIDES]
    for row in rows:
        result = pickle.loads(row['result'])
        for index, name in enumerate(SIDES):
            board = getattr(result, name).confirmed_board
            counts[index] += board is not None and board.get(*TARGET) != 0
    return [int(n) for n in counts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('off', type=Path)
    parser.add_argument('on', type=Path)
    parser.add_argument('baseline_off', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    off, on, base = read(args.off), read(args.on), read(args.baseline_off)
    assert [r['frame'] for r in off] == [r['frame'] for r in base]
    result = dict(scope='保存画素128コマ・10Hz・冷起動。eg短区間の代用にはならない。',
        baseline_off_mismatches={key: sum(a[key] != b[key] for a, b in zip(off, base))
            for key in ('result', 'published', 'internal', 'display')},
        off_on=compare(off, on), landing=dict(off=first_landing(off), on=first_landing(on)))
    offset, sets = 0, {}
    for name, count in SETS:
        sets[name] = compare(off[offset:offset+count], on[offset:offset+count])
        sets[name]['positive_top_off_1p_2p'] = positive_counts(off[offset:offset+count])
        sets[name]['positive_top_on_1p_2p'] = positive_counts(on[offset:offset+count])
        offset += count
    assert offset == len(off)
    result['sets'] = sets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
