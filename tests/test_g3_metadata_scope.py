"""原collector/Sink/Tailを使い、区間接続・非干渉・失敗保存を確認する。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import g3_metadata_scope as S

sys.path.insert(0, str(S.G.ROOT / 'data/verify/g2_collector_metadata_bounded_2026-09-09_v1'))
import test_bounded as T
sys.path.insert(0, str(S.G.ROOT / 'data/verify/g2_empty_tail_desync_trigger_2026-09-11_v1'))
import metadata_tail as M

collector = T.collector
B = T.B
FIXTURE_BOUNDS = (29052, 36298, 2, 60)


def calls(c: Any, frames: tuple[int, ...]) -> dict:
    """人工盤面で実採録関数へ連続入力する。"""
    acc, shared = c._LeanNpzAccumulator(), c._SharedGameCounter()
    states = {side: c._SideState() for side in B.O.SIDES}
    for frame in frames:
        for side in B.O.SIDES:
            T.call(c, acc, states[side], shared, frame, side)
    return {f.name: B.O.serial(getattr(acc, f.name)) for f in fields(acc) if f.name != 'wons'}


def test_actual_producer_tail_and_noninterference(collector: Any, tmp_path: Path) -> None:
    expected = calls(collector, (0, 2))
    state = dict(output=tmp_path)
    original = collector._process_side_lean
    with ExitStack() as outer:
        with ExitStack() as inner:
            B.install(inner, collector, None, state, enabled=True)
            S.bind_metadata(outer, state, FIXTURE_BOUNDS)
            tail = M.install(inner, state)
            assert calls(collector, (0, 2)) == expected
            tail.check(2)
        assert state[B.KEY].closed and B.O.FIRST == 0
        # prefixだけでは全区間保存合格にしない。元finisherの欠測拒否を保持。
        with pytest.raises(ValueError, match='metadata_saved_missing_rows'):
            B.finish(state)
        rows = [json.loads(line) for line in (tmp_path / B.O.ENTRIES).read_text().splitlines()]
        count = 0
        for index, row in enumerate(rows):
            count = B.O.validate_row(row, index // 2 * 2, B.O.SIDES[index % 2], count)
        assert len(rows) == 4
    assert collector._process_side_lean is original
    assert tuple(vars(B.O)[key] for key in S.NAMES) == FIXTURE_BOUNDS
    receipt = json.loads((tmp_path / 'G3_METADATA_SCOPE.json').read_text())
    assert receipt['restored'] and receipt['sink_closed'] and receipt['final_rows'] == 4
    assert receipt['error'] is None


@pytest.mark.parametrize('bad_frame', [2, 29052])
def test_initial_gap_refused(collector: Any, tmp_path: Path, bad_frame: int) -> None:
    state = dict(output=tmp_path)
    with pytest.raises(ValueError, match='metadata_saved_row_clock_or_schema'):
        with ExitStack() as outer, ExitStack() as inner:
            B.install(inner, collector, None, state, enabled=True)
            S.bind_metadata(outer, state, FIXTURE_BOUNDS)
            calls(collector, (bad_frame,))
    assert state[B.KEY].closed and B.O.FIRST == FIXTURE_BOUNDS[0]
    receipt = json.loads((tmp_path / 'G3_METADATA_SCOPE.json').read_text())
    assert receipt['restored'] and receipt['error']['type'] == 'ValueError'


def test_duplicate_refused(collector: Any, tmp_path: Path) -> None:
    state = dict(output=tmp_path)
    with pytest.raises(ValueError, match='metadata_saved_row_clock_or_schema'):
        with ExitStack() as outer, ExitStack() as inner:
            B.install(inner, collector, None, state, enabled=True)
            S.bind_metadata(outer, state, FIXTURE_BOUNDS)
            calls(collector, (0, 0))
    assert state[B.KEY].rows.count == 2 and state[B.KEY].closed


@pytest.mark.parametrize('reason', ['original_failure', S.G.CPU_STOP])
def test_exception_preserved(collector: Any, tmp_path: Path, reason: str) -> None:
    state = dict(output=tmp_path)
    error = RuntimeError(reason)
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as outer, ExitStack() as inner:
            B.install(inner, collector, None, state, enabled=True)
            S.bind_metadata(outer, state, FIXTURE_BOUNDS)
            raise error
    assert caught.value is error and state[B.KEY].closed
    receipt = json.loads((tmp_path / 'G3_METADATA_SCOPE.json').read_text())
    assert receipt['error']['message'] == reason and receipt['restored']
    assert receipt['planned_stop'] == (reason == S.G.CPU_STOP)


def test_unexpected_runtime_bounds_refused(collector: Any, tmp_path: Path) -> None:
    state = dict(output=tmp_path)
    with ExitStack() as outer, ExitStack() as inner:
        B.install(inner, collector, None, state, enabled=True)
        with pytest.raises(ValueError, match='metadata_original_bounds'):
            S.bind_metadata(outer, state)
    assert B.O.FIRST == FIXTURE_BOUNDS[0] and state[B.KEY].closed


def test_instance_append_wrapper_does_not_change_owner(collector: Any, tmp_path: Path) -> None:
    state = dict(output=tmp_path)
    with ExitStack() as outer, ExitStack() as inner:
        B.install(inner, collector, None, state, enabled=True)
        tail = M.install(inner, state)
        S.bind_metadata(outer, state, FIXTURE_BOUNDS)
        calls(collector, (0,))
        tail.check(0)
    assert state[B.KEY].rows.count == 2 and state[B.KEY].closed
