"""既存の別Belief/原alias-loader fixtureで実公開Session型の不変と復元を検査する。"""
from contextlib import ExitStack
from pathlib import Path
import sys
from typing import Any
from test_runtime_loader import test_private_belief_identity_through_loader as prepare
import session_runtime_binding as B


def test_existing_private_loader_and_real_session_class(monkeypatch: Any) -> None:
    prepare(monkeypatch)
    monkeypatch.setattr(B, 'physics', lambda module: (module.B.B.Board, module.B.B.ChainSimulator))
    loader = sys.modules['_publication_test_original_loader']
    module = sys.modules['_g2_pub_runtime_live_session']
    cls, witness_module = module.Session, module.W
    original_init, original_completed, original_load = cls.__init__, cls.completed, loader.load
    def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
        old = getattr(owner, name)
        stack.callback(setattr, owner, name, old)
        setattr(owner, name, value)
    with ExitStack() as stack:
        state = B.install(stack, loader, replace)
        loaded = loader.load('_g2_pub_runtime_live_session', Path(module.__file__))
        assert loaded is module and module.Session is cls
        assert state['binding'] is None  # ベース型への早期patchは禁止。
        patch = loader.load('_runtime_binding_actual_patch', B.PATCH)
        selected = patch.scheduled(loader.load, cls)
        replace(stack, module, 'Session', selected)  # 元runtime_patchのfactory後の置換点。
        binding = state['binding']
        assert binding['reader'].W is witness_module
        assert binding['consumer'].R is binding['reader']
        assert binding['cls'] is selected and cls.__init__ is original_init
        assert loader.load('_g2_pub_runtime_live_session', Path(module.__file__)) is module
        unrelated = sys.modules['_g2_pub_runtime_second_basis']
        assert loader.load('_g2_pub_runtime_second_basis', Path(unrelated.__file__)) is unrelated
    assert module.Session is cls and cls.__init__ is original_init
    assert cls.completed is original_completed and loader.load is original_load
    assert not any(alias in sys.modules for alias in B.ALIASES)
