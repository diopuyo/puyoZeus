"""原Sessionの生成/完了/保存/解放codeを検査。J witness・basis・モデルは人工の局所CPU。"""
from __future__ import annotations
import ast
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import whole_collector_capture as W
import whole_session_driver as D
from test_whole_collector_capture import REPO, C, fixture


def session_type(events: list) -> type:
    path = REPO / 'data/verify/g2_belief_live_publication_2026-09-11_v1/live_session.py'
    tree = ast.parse(path.read_bytes())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ('Session', 'write')]
    def install(stack: Any, *args: Any) -> Any:
        value = N(closed=False, error=None)
        stack.callback(setattr, value, 'closed', True)
        return value
    def save(current: Any, members: Any, path: Path, *, seed: int) -> dict:
        events.append(('evaluation', seed))
        return dict(frame=seed, artificial_model=True)
    namespace = dict(Any=Any, Path=Path, json=json, C=N(require=W.require),
                     W=N(install=install), O=N(install=install), SAVE=N(run=save))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Session']


def configuration(monkeypatch: Any, path: Path, selected: tuple = (102,)) -> tuple:
    collector, state, cls, shared, pipe, tracker, result, consumer, events = fixture(monkeypatch, path)
    original = session_type(events)
    state.update(probabilistic_basis_connection=N(), probabilistic_tracking_mode=N(),
                 provisional_context_observer=N(active=None, errors=[], rows=[dict(frame_idx=100)]))
    def create(scope: Any, context: Any) -> Any:
        value = original(scope, context, None, None, None, None, selected)
        # basis/M1品質は非対象。原completed/closeの呼出と保存条件だけ実行する。
        value.basis = lambda: setattr(value, 'mode', N(closed=False))
        return value
    driver = D.Driver(state, create, selected)
    def consume(bridge: Any, frame: int) -> None:
        state['provisional_context_observer'].rows = [dict(frame_idx=frame)]
        driver(bridge, frame)
    return collector, state, cls, shared, pipe, tracker, result, consume, events, driver


def test_original_session_close_after_driver_stop(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events, driver = configuration(monkeypatch, tmp_path)
    with ExitStack() as stack:
        bridge = W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
        collector.collect_lean(shared, pipe, tracker, result)
        assert driver.closed and driver.stopped
        assert bridge.consumer is W.forbidden_collect
        assert driver.session.restored and driver.session.evidence.closed and driver.session.witness.closed
    packet = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_text())
    assert packet['error'] is None and packet['session_error'] is None
    assert [row['frame'] for row in packet['saved']] == [102]
    assert events[-1] == ('evaluation', 102)


def test_install_frame_is_not_evaluation(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events, driver = configuration(monkeypatch, tmp_path, (100,))
    with pytest.raises(ValueError, match='session_started_too_late'):
        with ExitStack() as stack:
            W.Bridge(collector, state, stack, cls, C, (100, 102), consumer)
            collector.collect_lean(shared, pipe, tracker, result)
    assert driver.closed and driver.session is None


def test_no_mixed_old_driver() -> None:
    with pytest.raises(ValueError, match='duplicate_session_driver'):
        D.Driver(dict(belief_collector_boundary=object()), lambda *args: None, (102,))


def test_original_session_failure_still_restores(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events, driver = configuration(monkeypatch, tmp_path)
    def failing(bridge: Any, frame: int) -> None:
        if frame == 102:
            state['provisional_context_observer'].errors.append('original_observer_error')
        consumer(bridge, frame)
    with pytest.raises(ValueError, match='session_after_original_update'):
        with ExitStack() as stack:
            W.Bridge(collector, state, stack, cls, C, (100, 102), failing)
            collector.collect_lean(shared, pipe, tracker, result)
    assert driver.closed and driver.stopped and driver.session.restored
    assert driver.session.witness.closed and driver.session.evidence.closed
    assert (tmp_path / 'JOINT_PRODUCER_CAPTURE.json').exists()
    assert 'session_after_original_update' in json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_text())['error']
    assert 'session_after_original_update' in json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_text())['session_error']


def test_nonterminal_original_body_reaches_session(monkeypatch: Any, tmp_path: Path) -> None:
    collector, state, cls, shared, pipe, tracker, result, consumer, events, driver = configuration(monkeypatch, tmp_path)
    def failure(bridge: Any, frame: int) -> None:
        consumer(bridge, frame)
        if frame == 100:
            raise LookupError('original_nonterminal_body')
    with pytest.raises(LookupError, match='original_nonterminal_body'):
        with ExitStack() as stack:
            W.Bridge(collector, state, stack, cls, C, (100, 102), failure)
            collector.collect_lean(shared, pipe, tracker, result)
    packet = json.loads((tmp_path / 'BELIEF_M1_SESSION.json').read_text())
    assert 'original_nonterminal_body' in packet['error']
    original = json.loads((tmp_path / 'JOINT_PRODUCER_CAPTURE.failure.json').read_text())
    assert original['error_type'] == 'LookupError' and original['completed_frames'] == [100]
