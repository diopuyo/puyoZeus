"""正式版5c7c6333のnative指紋。APIだけ一致する別版を採用しない。"""

from __future__ import annotations

import hashlib

import json

from functools import cache

from pathlib import Path

from typing import Any

ROWS, COLS = 13, 6

COLORS = [1, 2, 3, 4]

VISIBLE = [(1, 2), (3, 4), (2, 2)]

AMA_WIDTH = 1

def boards() -> list[list[int]]:
    """決定的な試験盤 3 面 (空・4 連結 1 つ・2 連鎖の種)。"""
    empty = [0]*(ROWS*COLS)
    four = list(empty)
    four[-4:] = [1]*4
    two = list(empty)
    for col, color in ((0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (3, 2)):
        row = ROWS-1 if color == 1 else ROWS-2
        two[row*COLS+col] = color
    two[(ROWS-1)*COLS+3] = 1
    return [empty, four, two]

def fingerprint(native: Any) -> dict:
    """連鎖 trace・設置列挙・ama の順位を決定的な入力で呼び、結果の SHA を返す。"""
    out = {}
    for i, board in enumerate(boards()):
        trace = native.simulate_chain_trace_py(list(board), True)
        out[f'trace_{i}'] = hashlib.sha256(json.dumps([list(map(repr, s)) for s in trace]).encode()).hexdigest()
        rows = native.enumerate_and_simulate_placements_py(list(board), *VISIBLE[0], False, True)
        summary = [(int(r.exact_score), list(r.final_grid)) for r in rows]
        out[f'place_{i}'] = hashlib.sha256(json.dumps(summary).encode()).hexdigest()
        ranked = native.ama_search_py(list(board), VISIBLE, AMA_WIDTH, len(VISIBLE), COLORS)
        out[f'ama_{i}'] = hashlib.sha256(json.dumps([list(map(repr, r)) for r in ranked]).encode()).hexdigest()
    return out

@cache
def verify_native(native: Any) -> dict:
    expected = json.loads(Path(__file__).with_name('live_ex_native_golden.json').read_text(encoding='utf-8'))
    actual = fingerprint(native)
    if actual != expected['fingerprint']:
        mismatches = [name for name in actual if actual[name] != expected['fingerprint'].get(name)]
        raise ValueError(f'正式nativeの指紋と不一致: {len(mismatches)}/{len(actual)} {mismatches}')
    directory = Path(native.__file__).parent
    libraries = sorted(directory.glob('*.so')) + sorted(directory.glob('*.pyd'))
    return dict(fingerprint=actual, libraries={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries})
