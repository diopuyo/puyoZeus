"""実画像state有りの非干渉・逐次保存・メモリ保持を人工入力で検査する。"""
from __future__ import annotations
import ast
import contextlib
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import resource
import sys
from typing import Any
import numpy as np
import pytest
import bounded as B

sys.path.insert(0, str(B.PRIOR))
import test_observer as T
collector = T.collector
O = B.O
GRAY_SHAPE = (640, 480)
MEMORY_LIMIT_KIB = 64 * 1024


def call(c: Any, acc: Any, state: Any, shared: Any, frame: int, side: str) -> None:
    grid = [[0] * 6 for _ in range(13)]
    grid[12][0] = 1 if side == '1P' else 2
    c._process_side_lean(acc, state, side, c.Board.from_list(grid), c.BoardState.STABLE,
        90, 'artificial-bounded-metadata', frame / O.FPS, frame, next_pair=(1, 2),
        dnext_pair=(3, 4), shared_game=shared, tsumo_count=2, all_clear_pending=1,
        ojama_net_balance=-3.0, ojama_forecast=4.0, match_end_locked=False,
        post_match_lockdown_active=False, stable_persistence_confidence=True, board_provenance='observed')


def run(c: Any, output: Path, enabled: bool) -> tuple[Any, Any]:
    output.mkdir(exist_ok=False)
    state = {'output': output}
    acc, shared = c._LeanNpzAccumulator(), c._SharedGameCounter()
    gray = np.arange(np.prod(GRAY_SHAPE), dtype=np.uint8).reshape(GRAY_SHAPE)
    sides = {side: c._SideState(motion_prev_gray=gray.copy()) for side in O.SIDES}
    before = O.serial, c._process_side_lean
    with contextlib.ExitStack() as stack:
        B.install(stack, c, None, state, enabled=enabled)
        for frame in range(O.FIRST, O.LAST + O.STRIDE, O.STRIDE):
            for side in O.SIDES:
                call(c, acc, sides[side], shared, frame, side)
    assert before == (O.serial, c._process_side_lean)
    assert all(np.array_equal(v.motion_prev_gray, gray) for v in sides.values())
    saved = {f.name: O.serial(getattr(acc, f.name)) for f in fields(acc) if f.name != 'wons'}
    return saved, state


def test_gray_noninterference_and_exact_saved_rows(collector: Any, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(O, 'LAST', O.FIRST + O.STRIDE)
    old, _ = run(collector, tmp_path / 'off', False)
    new, state = run(collector, tmp_path / 'on', True)
    assert old == new and set(new) == O.STORED_KEYS
    B.finish(state)
    B.verify(tmp_path / 'on')
    rows = [json.loads(line) for line in (tmp_path / 'on' / O.ENTRIES).read_text().splitlines()]
    assert len(rows) == 4 and sum(len(r['appends']) for r in rows) == 2
    for row in rows:
        image = row['state_before']['fields'][B.GRAY]
        assert image['excluded_image_payload'] is True and image['shape'] == list(GRAY_SHAPE)
        assert image == row['state_after']['fields'][B.GRAY] and 'values' not in image


def test_long_stream_has_no_accumulated_rows(collector: Any, tmp_path: Path, monkeypatch: Any) -> None:
    updates = 1000
    monkeypatch.setattr(O, 'LAST', O.FIRST + (updates - 1) * O.STRIDE)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _, state = run(collector, tmp_path / 'long', True)
    sink = state[B.KEY]
    growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert set(vars(sink.rows)) == {'stream', 'count', 'append_count'}
    assert sink.rows.count == updates * len(O.SIDES) and sink.rows.append_count == 2
    assert growth < MEMORY_LIMIT_KIB
    B.finish(state)
    B.verify(tmp_path / 'long')
    O.write(tmp_path / 'MEMORY.json', {'artificial_updates': updates, 'gray_shape': GRAY_SHAPE,
        'maxrss_growth_kib': growth, 'retained_row_list': False, 'quality_gate_clear': False})


def test_default_off_and_descriptor_refuse() -> None:
    with contextlib.ExitStack() as stack:
        B.install(stack, object(), None, {}, enabled=False)
    for value in (np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.float64)):
        with pytest.raises(ValueError, match='gray_shape_type'):
            B.image_descriptor(value)


def test_function_lengths() -> None:
    for name in B.OWN:
        if name.endswith('.py'):
            for node in ast.walk(ast.parse((B.ROOT / name).read_bytes())):
                if isinstance(node, ast.FunctionDef):
                    assert node.end_lineno - node.lineno + 1 <= 50, (name, node.name)
