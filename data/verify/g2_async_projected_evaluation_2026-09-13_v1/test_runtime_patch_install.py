"""元runtime_patch.installの中間subclass/生成/常時guardを実経路で通す。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as N
import sys
import json
from typing import Any
import pytest
from test_runtime_loader import test_private_belief_identity_through_loader as prepare
import session_runtime_binding as B
from test_journal_pair_reader import setup


def construct_and_close(module: Any, arrival: Any, state: dict, output: Path) -> None:
    """物理Mode初期状態は人工。元Session/既存Witness/終了保存は実実装を通す。"""
    journal, pipe, _ = setup()
    first = arrival.Mode()
    first.connection = N(binding=N(scope=('artificial', '1P')))
    context = dict(pipe=pipe, factory=N(provider=N(journal=journal)),
                   state=dict(output=output, probabilistic_tracking_mode=first))
    targets = module.Session.__init__.__wrapped__.__globals__['S'].EARLIEST
    planned = RuntimeError('planned_CPU_stop_after_original_Session_constructor')
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            value = module.Session(stack, context, None, N(Mode=arrival.Mode), None, None, targets)
            B.verify(state, require_instance=True)
            assert type(value.witness) is state['binding']['reader'].W.Witness
            assert state['binding']['created_frames'] == [journal.history.frame]
            raise planned
    assert caught.value is planned and value.witness.closed and value.evidence.closed
    assert value.evaluation_flags.closed and value.schedule_stream.closed
    receipt = json.loads((output / 'PROJECTED_ORIGIN_CAPTURE_CLOSE.json').read_text())
    assert receipt['capture_closed'] and receipt['stream_closed']
    assert receipt['original_error'] == repr(planned)
    assert json.loads((output / 'BELIEF_M1_SESSION.json').read_text())['error'] == repr(planned)


def test_original_runtime_patch_install_and_class_guard(monkeypatch: Any, tmp_path: Path) -> None:
    prepare(monkeypatch)
    monkeypatch.setattr(B, 'physics', lambda module: (module.B.B.Board, module.B.B.ChainSimulator))
    original_loader = sys.modules['_publication_test_original_loader']
    module = sys.modules['_g2_pub_runtime_live_session']
    base_cls, base_completed = module.Session, module.Session.completed
    shared = N(require=module.C.require)
    second = N(Mode=type('Second', (), {}), B=shared)
    arrival = N(Mode=type('First', (second.Mode,), {}), BASE=second, B=shared)
    monkeypatch.setitem(sys.modules, '_g2_real_basis_cascade_mode_v2', second)
    arrival_path = B.ROOT.parent / 'g2_arrival_ack_candidate_2026-09-12_v1/arrival_mode.py'
    def raw_load(alias: str, path: Path, injection: Any = None) -> Any:
        if Path(path).resolve() == arrival_path: return arrival
        return original_loader.load(alias, path, injection)
    bootstrap = N(load=raw_load)
    def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
        old = getattr(owner, name)
        stack.callback(setattr, owner, name, old)
        setattr(owner, name, value)
    with ExitStack() as stack:
        state = B.install(stack, bootstrap, replace)
        with pytest.raises(ValueError, match='patch_not_hooked'): B.verify(state)
        patch = bootstrap.load('_actual_install_runtime_patch', B.PATCH)
        with pytest.raises(ValueError, match='session_not_bound'): B.verify(state)
        original_state = patch.install(stack, bootstrap, replace)
        bootstrap.load('_actual_install_arrival_fixture', arrival_path)
        assert bootstrap.load('_g2_pub_runtime_live_session', Path(module.__file__)) is module
        B.verify(state)
        with pytest.raises(ValueError, match='instance_not_created'):
            B.verify(state, require_instance=True)
        assert original_state['session'][1] is state['binding']['cls'] is module.Session
        assert module.Session.__mro__[2] is base_cls and base_cls.completed is base_completed
        bootstrap.load('_actual_install_runtime_patch', B.PATCH)  # 元常時identity guardも実行。
        construct_and_close(module, arrival, state, tmp_path)
    assert module.Session is base_cls and base_cls.completed is base_completed
    assert bootstrap.load is raw_load and not any(alias in sys.modules for alias in B.ALIASES)
