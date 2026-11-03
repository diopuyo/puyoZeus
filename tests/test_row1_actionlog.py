"""作動ログの全分岐・影モード・判定非干渉を確認する。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.board import Board
from src.phase_j.live_audit import RecognitionAudit
from src.phase_j.live_row1_audit import ActionLog, CURRENT, action_scope
from src.recognition_pipeline import RecognitionPipeline
from src.row1_guard import Row1Guard


def board(top: int = 0, hidden: int = 0, below: int = 2) -> Board:
    result = Board()
    for row, value in enumerate((hidden, top, below)):
        result.set(row, 2, value)
    return result


@pytest.mark.parametrize('branch', ['floating', 'screen_empty', 'spawn_held', 'released'])
def test_each_branch_emits_one_line(tmp_path: Path, branch: str) -> None:
    guard = Row1Guard()
    log = ActionLog(tmp_path/'actions.jsonl')
    empty, spawn = board(), board(1, 3)
    if branch in ('spawn_held', 'released'):
        guard.step(empty, empty, 0.)
    if branch == 'released':
        log.apply(guard, spawn, spawn, .1, False, 1, '2P', False)
    if branch == 'screen_empty':
        guard.step(spawn, empty, 0.)
    candidate = board(1, below=0) if branch == 'floating' else spawn
    count = len(log.path.read_text().splitlines())
    raw = candidate if branch in ('spawn_held', 'released') else empty
    log.apply(guard, candidate, raw, .5, branch == 'released', 5, '2P', False)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert len(rows) == count + 1
    row = rows[-1]
    assert (row['kind'], row['frame'], row['t_sec'], row['side']) == (branch, 5, .5, '2P')
    assert row['before'] == {'row0_col2': candidate.get(0, 2), 'row1_col2': 1}
    if branch == 'released':
        assert row['hold_id'] == rows[0]['hold_id']
        assert row['release_reason'] == 'chain_started'


@pytest.mark.parametrize('reason', ['next_moved', 'ojama_falling', 'raw_disappeared',
                                   'top_empty', 'no_board'])
def test_other_hold_releases(tmp_path: Path, reason: str) -> None:
    guard, log = Row1Guard(), ActionLog(tmp_path/'actions.jsonl')
    empty, spawn = board(), board(1, 3)
    guard.step(empty, empty, 0.)
    log.apply(guard, spawn, spawn, .1, False, 1, '1P', False)
    value = {'top_empty': empty, 'no_board': None}.get(reason, spawn)
    raw = empty if reason == 'raw_disappeared' else spawn
    log.apply(guard, value, raw, .7, False, 7, '1P', False,
              next_moved=reason == 'next_moved', ojama_falling=reason == 'ojama_falling')
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]['hold_start'] and rows[0]['hold_id'] == rows[1]['hold_id']
    assert rows[1]['release_reason'] == reason


def test_screen_suppression_and_restore(tmp_path: Path) -> None:
    guard, log = Row1Guard(), ActionLog(tmp_path/'actions.jsonl')
    empty, full = board(), board(1)
    for index, (raw, time) in enumerate(((empty, 0.), (empty, .5), (full, .6), (full, 1.2))):
        log.apply(guard, full, raw, time, False, index, '1P', False, next_moved=index == 3)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [r['reason'] for r in rows] == ['empty_evidence', 'suppression_continues', 'next_moved']
    assert rows[-1]['evidence']['restored'] and rows[-1]['hold_id'] is None


def test_same_frame_start_and_release(tmp_path: Path) -> None:
    guard, log = Row1Guard(), ActionLog(tmp_path/'actions.jsonl')
    guard.step(board(), board(), 0.)
    log.apply(guard, board(1, 3), board(1, 3), .1, True, 1, '1P', False)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['hold_start']
    assert rows[0]['hold_id'] is not None and rows[0]['release_reason'] == 'chain_started'


def test_disappeared_hold_closes_before_screen_restore(tmp_path: Path) -> None:
    guard, log = Row1Guard(), ActionLog(tmp_path/'actions.jsonl')
    empty, spawn = board(), board(1, 3)
    guard.step(empty, empty, 0.)
    sequence = ((spawn, .1), (empty, .2), (empty, .7), (spawn, .8), (spawn, 1.4), (spawn, 1.5))
    for index, (raw, time) in enumerate(sequence):
        log.apply(guard, spawn, raw, time, False, index, '1P', False, next_moved=index == 5)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    removed = next(r for r in rows if r['reason'] == 'raw_disappeared')
    assert removed['hold_id'] == rows[0]['hold_id']
    assert rows[-1]['hold_id'] is None and rows[-1]['evidence']['restored']
    assert rows[-1]['reason'] == 'next_moved'


def test_scope_cleans_up_on_exception(tmp_path: Path) -> None:
    audit = RecognitionAudit(tmp_path/'recognition.npz')
    with pytest.raises(RuntimeError), action_scope(audit, SimpleNamespace(index=1)):
        assert CURRENT.get() is not None
        raise RuntimeError('テスト用')
    assert CURRENT.get() is None


def test_sides_and_repeated_holds_have_unique_ids(tmp_path: Path) -> None:
    log = ActionLog(tmp_path/'actions.jsonl')
    for side in ('1P', '2P'):
        guard = Row1Guard()
        for index in (1, 2):
            guard.step(board(), board(), float(index))
            log.apply(guard, board(1, 3), board(1, 3), index + .1, False, index, side, False)
            log.apply(guard, None, None, index + .2, False, index, side, False)
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert len({r['hold_id'] for r in rows if r['hold_start']}) == 4
    assert all(rows[i]['hold_id'] == rows[i + 1]['hold_id'] for i in range(0, len(rows), 2))


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('auditing', [False, True])
def test_audit_matrix_preserves_outputs(tmp_path: Path, enabled: bool, auditing: bool) -> None:
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._enable_row1_col2_guard, pipe._row1_guards = enabled, {}
    baseline = Row1Guard()
    audit = RecognitionAudit(tmp_path/'recognition.npz' if auditing else None)
    empty, spawn, floating = board(), board(1, 3), board(1, below=0)
    sequence = [(empty, empty, False), (spawn, spawn, False), (spawn, spawn, True),
                (floating, empty, False)] + [(spawn, empty, False)] * 10
    for index, (candidate, raw, chain) in enumerate(sequence):
        original = candidate._grid.tobytes()
        expected = baseline.step(candidate, raw, index / 10, chain) if enabled else candidate
        with action_scope(audit, SimpleNamespace(index=index)):
            actual = pipe._apply_row1_col2_guard('1P', candidate, raw, index / 10, chain)
        assert actual._grid.tobytes() == expected._grid.tobytes()
        assert candidate._grid.tobytes() == original
        assert CURRENT.get() is None
    path = tmp_path/'recognition.row1_guard.jsonl'
    assert path.exists() == auditing
    if auditing:
        assert all(r['shadow'] == (not enabled) for r in map(json.loads, path.read_text().splitlines()))
    if not enabled:
        assert pipe.row1_guard_counts() == {}


@pytest.mark.parametrize('signal', ['in_chain', 'next_moved', 'ojama_falling'])
@pytest.mark.parametrize('enabled', [False, True])
def test_pipeline_signals_are_side_local(tmp_path: Path, signal: str, enabled: bool) -> None:
    """実ガード・影の両方に合図を渡し、他方の側は保留を継続する。"""
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._enable_row1_col2_guard, pipe._row1_guards = enabled, {}
    audit = RecognitionAudit(tmp_path/'recognition.npz')
    empty, spawn = board(), board(1, 3)
    for index, candidate in enumerate((empty, spawn, spawn)):
        with action_scope(audit, SimpleNamespace(index=index)):
            for side in ('1P', '2P'):
                signals = {signal: index == 2 and side == '1P'}
                out = pipe._apply_row1_col2_guard(side, candidate, candidate, index / 30, **signals)
                if index == 2:
                    assert out.get(1, 2) == (1 if side == '1P' or not enabled else 0)
    rows = list(map(json.loads, (tmp_path/'recognition.row1_guard.jsonl').read_text().splitlines()))
    assert rows[-2]['kind'] == 'released' and rows[-2]['side'] == '1P'
    assert rows[-1]['kind'] == 'spawn_held' and rows[-1]['side'] == '2P'
    assert rows[-2]['evidence']['raw_top'] == rows[-2]['evidence']['after_top'] == 1


def test_formula_input_is_current_side_reading() -> None:
    pipe = RecognitionPipeline.__new__(RecognitionPipeline)
    pipe._formula_last_read_1p = SimpleNamespace(valid=True)
    pipe._formula_last_read_2p = SimpleNamespace(valid=False)
    assert pipe._row1_formula_visible('1P') and not pipe._row1_formula_visible('2P')
    pipe._formula_last_read_1p = None
    assert not pipe._row1_formula_visible('1P')
