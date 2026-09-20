"""元scheduled factoryが参照するcompleted globalsを、早期patchで壊さない。"""
from contextlib import ExitStack
from pathlib import Path
import sys
from typing import Any
from test_runtime_loader import test_private_belief_identity_through_loader as prepare
import session_runtime_binding as B


def test_original_scheduled_factory_after_publication_load(monkeypatch: Any) -> None:
    prepare(monkeypatch)
    monkeypatch.setattr(B, 'physics', lambda module: (module.B.B.Board, module.B.B.ChainSimulator))
    loader = sys.modules['_publication_test_original_loader']
    module = sys.modules['_g2_pub_runtime_live_session']
    def replace(stack: Any, owner: Any, name: str, value: Any) -> None:
        old = getattr(owner, name)
        stack.callback(setattr, owner, name, old)
        setattr(owner, name, value)
    with ExitStack() as stack:
        state = B.install(stack, loader, replace)
        path = B.ROOT.parent / 'g2_m1_completion_candidate_2026-09-12_v1/runtime_patch.py'
        patch = loader.load('_factory_order_actual_runtime_patch', path)
        original = module.Session
        loader.load('_g2_pub_runtime_live_session', Path(module.__file__))
        selected = patch.scheduled(loader.load, original)
        assert selected is not original
        assert state['binding']['cls'] is selected
        assert module.Session is original  # 元runtime_patchが後で置換する直前の生成点。
        bound_init = selected.__init__
    assert selected.__init__ is bound_init.__wrapped__
    assert not any(alias in sys.modules for alias in B.ALIASES)
