"""121指摘の依存寿命/原生成順/実anchor型を限定CPUで確認する。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as N
from typing import Any
import pytest
import whole_dependencies as D
import whole_collector_capture as W
from test_whole_collector_capture import REPO, C, fixture, extracted


def test_dependency_failure_restores_original(monkeypatch: Any) -> None:
    before, paths = dict(sys.modules), list(sys.path)
    def failed(load: Any) -> Any:
        sys.modules[D.ALIASES[0]] = ModuleType(D.ALIASES[0])
        sys.path.insert(0, 'artificial-owned-path')
        raise LookupError('original_partial_import')
    monkeypatch.setattr(D, 'observer_and_anchor', failed)
    with pytest.raises(LookupError, match='original_partial_import'):
        with ExitStack() as stack:
            scope = D.Scope(stack)
            scope.prepare(None, (100,))
    assert scope.closed and sys.path == paths and sys.modules == before


def test_foreign_alias_is_not_removed(monkeypatch: Any) -> None:
    foreign = ModuleType(D.ALIASES[0])
    monkeypatch.setitem(sys.modules, D.ALIASES[0], foreign)
    with ExitStack() as stack, pytest.raises(ValueError, match='whole_dependency_foreign_alias'):
        D.Scope(stack)
    assert sys.modules[D.ALIASES[0]] is foreign


def test_reversed_original_constructors_are_rejected(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    tree = extracted()
    body = tree.body[0].body
    body[0], body[1] = body[1], body[0]
    exec(compile(tree, 'artificial_reversed_constructor.py', 'exec'), vars(collector))
    with pytest.raises(ValueError, match='physical_constructor'):
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
            collector.collect_lean(shared, pipe, tracker, result)
    assert collector.EventAccountingRecorder is cls and bridge.capture is None


def anchor(monkeypatch: Any) -> Any:
    path = REPO / 'data/verify/g2_observed_start_anchor_2026-09-12_v1/anchor.py'
    spec = importlib.util.spec_from_file_location('_review121_anchor_base', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    monkeypatch.setitem(sys.modules, 'anchor', base)
    path = REPO / 'data/verify/g2_joint_collector_runtime_2026-09-12_v1/anchor_v2.py'
    spec = importlib.util.spec_from_file_location('_review121_anchor_v2', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def enrich(state: dict, pipe: Any, result: Any, *, missing: bool) -> None:
    tail = state['reset_metadata_tail']
    previous = tail.append
    def append(frame: int, side: str) -> None:
        previous(frame, side)
        tail.rows[-1] = dict(frame_idx=frame, side=side, arguments=dict(board=None,
            bstate=dict(value='menu'), score=0, board_provenance='observed', chain_event_used_fields=None,
            raw_pixel_stable=False, stable_persistence_confidence=False))
        if missing and side == '2P':
            tail.rows[:] = tail.rows[-1:]
    def check(frame: int) -> None:
        W.require([(r['frame_idx'], r['side']) for r in tail.rows] == [(frame, s) for s in ('1P', '2P')], 'actual_anchor_pair')
    tail.append, tail.check = append, check
    result.is_match_active = True
    for side in (result.p1, result.p2):
        side.state, side.score, side.board_provenance = N(value='menu'), 0, 'observed'
    pipe._active_chain_1p = pipe._active_chain_2p = None


@pytest.mark.parametrize('missing', [False, True])
def test_real_anchor_missing_board_path(monkeypatch: Any, tmp_path: Path, missing: bool) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, tmp_path)
    actual = anchor(monkeypatch)
    enrich(state, pipe, result, missing=missing)
    def run() -> Any:
        with ExitStack() as stack:
            bridge = W.Bridge(collector, state, stack, cls, actual, (100, 102), consumer)
            collector.collect_lean(shared, pipe, tracker, result)
        return bridge
    if missing:
        with pytest.raises(ValueError, match='actual_anchor_pair'):
            run()
    else:
        bridge = run()
        assert bridge.capture.closed and bridge.capture.anchor is None
        assert len(bridge.capture.unavailable_boards) == 4
