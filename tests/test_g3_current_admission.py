"""実collectorのappend/同frame保留と元writer失敗を小CPUで確認する。"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
from scripts import g3_current_admission as G
from tests.test_g3_admission import A, T, collector, pipeline, owned_replace, admission_root  # noqa: F401  admission_root は autouse fixture。採録先の根を一時領域へ差し替える (W45)
from tests.test_g3_current_exit import result
from tests.test_g3_live_current_guard import live

FRAME = 29052


def matching_result(c: Any) -> Any:
    """既存T.call人工盤面と一致する元結果型。人手GTではない。"""
    value = result(FRAME)
    sides = {}
    for side, name in G.C.SIDES:
        grid = [[0] * 6 for _ in range(13)]
        grid[12][0] = 1 if side == '1P' else 2
        sides[name] = replace(getattr(value, name), confirmed_board=c.Board.from_list(grid))
    return replace(value, **sides)


@pytest.mark.parametrize('mode', ['normal', 'reset', 'unknown', 'old_no_gate'])
def test_actual_collector_gate(collector: Any, monkeypatch: Any, tmp_path: Path, mode: str) -> None:
    """元_should_emit→Admission.decide→実appendの経路を変えずに保留する。"""
    c = collector
    pipe, ocr = pipeline(c, monkeypatch)
    state, acc, shared = dict(output=tmp_path), c._LeanNpzAccumulator(), c._SharedGameCounter()
    before = (G.C.selection, G.C.CurrentExit.consume, A.Admission.decide, c._should_emit)
    with ExitStack() as stack:
        observer = A.install(stack, c, state, owned_replace(), source_id='artificial-current-admission')
        current = G.C.install(stack, c, state, owned_replace())
        G.L.install(stack, state, owned_replace())
        gate = None if mode == 'old_no_gate' else G.install(stack, state, owned_replace())
        if gate:
            assert not gate.permitted(FRAME, FRAME / 60)
            with pytest.raises(ValueError, match='install_order'):
                G.install(stack, state, owned_replace())
        value = matching_result(c)
        live(pipe, value)
        if mode in ('reset', 'old_no_gate'):
            pipe._sm_2p.reset(keep_match_state=False)
        ocr.values['1P'] = None if mode == 'unknown' else 0
        assert pipe.update(FRAME, FRAME / 60, value) is value
        for side in A.SIDES:
            T.call(c, acc, c._SideState(), shared, FRAME, side)
        assert len(acc.grids) == (2 if mode in ('normal', 'old_no_gate') else 0)
        assert observer.decisions == 2
        if gate:
            assert not gate.permitted(FRAME + 2, (FRAME + 2) / 60)
            assert observer.decide(FRAME, FRAME / 60, '1P', False) is False
    rows = [json.loads(x) for x in (tmp_path / 'G3_UI_ADMISSION.jsonl').read_text().splitlines()]
    decisions = [x for x in rows if x['kind'] == 'admission']
    assert decisions[0]['original_eligible'] and decisions[1]['original_eligible']
    assert current.stream.closed and before == (G.C.selection, G.C.CurrentExit.consume, A.Admission.decide, c._should_emit)


@pytest.mark.parametrize('sink', ['current', 'admission'])
def test_saved_failure_latches(collector: Any, monkeypatch: Any, tmp_path: Path, sink: str) -> None:
    """保存失敗を捕捉されても次の採録資格へ戻さない。"""
    pipe, ocr = pipeline(collector, monkeypatch)
    state = dict(output=tmp_path)
    with ExitStack() as stack:
        observer = A.install(stack, collector, state, owned_replace(), source_id='artificial-save-failure')
        current = G.C.install(stack, collector, state, owned_replace())
        G.L.install(stack, state, owned_replace())
        gate = G.install(stack, state, owned_replace())
        value = matching_result(collector)
        live(pipe, value)
        ocr.values['1P'] = 0
        def fail(*args: Any) -> Any:
            raise OSError('current_admission_write_failed')
        if sink == 'current':
            monkeypatch.setattr(G.C, 'encoded', fail)
            with pytest.raises(OSError, match='write_failed'):
                pipe.update(FRAME, FRAME / 60, value)
        else:
            pipe.update(FRAME, FRAME / 60, value)
            monkeypatch.setattr(observer, 'write', fail)
            with pytest.raises(OSError, match='write_failed'):
                observer.decide(FRAME, FRAME / 60, '1P', True)
        assert gate.latest is None and gate.error and not gate.permitted(FRAME, FRAME / 60)
    assert current.stream.closed and observer.closed


def test_foreign_owner_passthrough(collector: Any, monkeypatch: Any, tmp_path: Path) -> None:
    """別所有者への直接呼出は元処理へ一回委ねる。"""
    pipe, _ = pipeline(collector, monkeypatch)
    state = dict(output=tmp_path)
    with ExitStack() as stack:
        A.install(stack, collector, state, owned_replace(), source_id='artificial-foreign')
        G.C.install(stack, collector, state, owned_replace())
        G.L.install(stack, state, owned_replace())
        gate = G.install(stack, state, owned_replace())
        other, calls = N(), []
        def original(*args: Any) -> bool:
            calls.append(args)
            return True
        assert gate.decide(original, other, 1, 0.0, '1P', True)
        assert gate.consume(original, other) is True
        assert len(calls) == 2 and gate.latest is None
