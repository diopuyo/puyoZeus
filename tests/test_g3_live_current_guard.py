"""人工結果と実SMでreset反例、正常対照、既存保存/終了経路を確認する。"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
from scripts import g3_live_current_guard as L
from tests.test_g3_current_exit import result
from tests.test_g3_admission import A, collector, pipeline, owned_replace, admission_root  # noqa: F401  admission_root は autouse fixture。採録先の根を一時領域へ差し替える (W45)

FRAME = 8724


def live(pipe: Any, value: Any) -> None:
    """凍結実SMへ人工結果と同値のcopyを置く。画像/人手GTではない。"""
    from src.board_state_machine import BoardStateMachine
    for side, name in L.C.SIDES:
        sm = BoardStateMachine()
        sm.context.state = getattr(value, name).state
        sm.context.frame_idx = value.frame_idx
        sm.context.confirmed_board = getattr(value, name).confirmed_board.copy()
        setattr(pipe, '_sm_' + side.lower(), sm)


@pytest.mark.parametrize('mode', ['normal', 'reset', 'old_clock', 'different', 'missing', 'alias', 'hidden'])
def test_live_context_boundary(collector: Any, mode: str) -> None:
    """同値copyは通し、旧result/時計/side混線を拒否。隠し段は観測比較しない。"""
    value, pipe = result(FRAME), N()
    live(pipe, value)
    ctx = pipe._sm_2p.context
    if mode == 'reset':
        pipe._sm_2p.reset(keep_match_state=False)
    elif mode == 'old_clock':
        ctx.frame_idx -= 2
    elif mode in ('different', 'hidden'):
        ctx.confirmed_board.set(1 if mode == 'different' else 0, 0, 2)
    elif mode == 'missing':
        del pipe._sm_2p
    elif mode == 'alias':
        pipe._sm_2p = pipe._sm_1p
    observer = N(pipe=pipe, active=None, error=None, save_error=None,
                 latest=dict(frame=FRAME, time_sec=FRAME / 60, status=A.OBSERVED))
    before = value.p2.confirmed_board._grid.copy()
    row = L.wrap(L.C.selection)(observer, pipe, value, FRAME, FRAME / 60)
    good = mode in ('normal', 'hidden')
    assert (row['status'] == 'CURRENT_INPUT_ELIGIBLE') == good
    assert row['live_context_eligible'] == good
    assert (row['visible_confirmed_boards'] is not None) == good
    assert (before == value.p2.confirmed_board._grid).all()


def test_saved_real_reset_hold(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """既存Admission/出口writerへ接続して2行保存し、元参照復元を確認する。"""
    pipe, ocr = pipeline(collector, monkeypatch)
    state, original = dict(output=tmp_path), L.C.selection
    with ExitStack() as stack:
        A.install(stack, collector, state, owned_replace(), source_id='artificial-live-context')
        current = L.C.install(stack, collector, state, owned_replace())
        L.install(stack, state, owned_replace())
        with pytest.raises(ValueError, match='duplicate'):
            L.install(stack, state, owned_replace())
        ocr.values['1P'] = 0
        value = result(0)
        live(pipe, value)
        assert pipe.update(0, 0.0, value) is value
        value = result(2)
        live(pipe, value)
        pipe._sm_2p.reset(keep_match_state=False)
        assert pipe.update(2, 2 / 60, value) is value
    rows = [json.loads(line) for line in (tmp_path / 'G3_CURRENT_EXIT.jsonl').read_text().splitlines()]
    assert [row['status'] for row in rows] == ['CURRENT_INPUT_ELIGIBLE', 'HOLD']
    assert current.rows == 2 and current.eligible == 1 and current.stream.closed
    assert rows[1]['visible_confirmed_boards'] is None and L.C.selection is original
    assert not any(row['human_gt'] or row['model_evaluation_completed'] for row in rows)


def test_guarded_save_failure(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """新ガード接続後も保存障害を握らず、元参照と閉鎖票を残す。"""
    pipe, ocr = pipeline(collector, monkeypatch)
    state, original = dict(output=tmp_path), L.C.selection
    with pytest.raises(OSError, match='guarded_write_failed'):
        with ExitStack() as stack:
            A.install(stack, collector, state, owned_replace(), source_id='artificial-live-failure')
            current = L.C.install(stack, collector, state, owned_replace())
            L.install(stack, state, owned_replace())
            ocr.values['1P'] = 0
            value = result(0)
            live(pipe, value)
            def fail(row: dict) -> bytes:
                raise OSError('guarded_write_failed')
            monkeypatch.setattr(L.C, 'encoded', fail)
            pipe.update(0, 0.0, value)
    status = json.loads((tmp_path / 'G3_CURRENT_EXIT_STATUS.json').read_text())
    assert current.rows == 0 and current.stream.closed and L.C.selection is original
    assert status['closed'] and status['references_restored'] and 'guarded_write_failed' in status['save_error']
