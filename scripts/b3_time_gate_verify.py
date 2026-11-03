"""計測器のフレームループを既存generate_rowsと短い同一prefixで照合する。"""
from __future__ import annotations

from pathlib import Path
import ast
import json

import pandas as pd

from src.b3_rows import generate_rows
from src.b3_source import iter_frames, resolve_attempt
from src.b3_timing import TimingTable
from scripts.b3_time_gate import ROOT, TABLE, Replay

PREFIX_FRAME = 5400


def main() -> None:
    video = 'b3_zenchi_s1'
    record = resolve_attempt(video, ROOT / 'b3_eval_records').record
    table = TimingTable.from_json(TABLE.read_text())
    replay = Replay(record, video, table)
    actual = []
    for frame in iter_frames(record, PREFIX_FRAME):
        row, _ = replay.step(frame)
        if row is not None:
            actual.append(row)
    mapping = {int(k): v for k, v in replay.mapping.items()}
    expected, _ = generate_rows(record, video, table, max_frame=PREFIX_FRAME, winners={},
                                require_winner=False, match_mapping=mapping)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), check_exact=True)
    lengths = {}
    for path in Path('scripts').glob('b3_time_gate*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lengths[f'{path.name}:{node.name}:{node.lineno}'] = node.end_lineno-node.lineno+1
    if max(lengths.values()) > 50:
        raise AssertionError({k: v for k, v in lengths.items() if v > 50})
    result = dict(prefix_frame=PREFIX_FRAME, rows=len(actual), exact_equal=True,
                  maximum_function_lines=max(lengths.values()))
    destination = ROOT / 'b3_data/time_gate/verification.json'
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
