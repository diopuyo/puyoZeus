"""元Session factoryにoptional履歴取得口が届くことを検査。履歴実体は別CPU検収。"""
from types import SimpleNamespace as N
from typing import Any
import session_runtime_binding as B
from test_projected_binding_scope import test_projected_callback_binding_scope_close as scenario


def test_history_getter_reaches_original_factory(monkeypatch: Any, tmp_path: Any) -> None:
    calls: list[int] = []
    original = B.install
    def prime(capture: Any, frame: int) -> None:
        assert capture.last_frame == -1 and capture.projected_input is not None
        calls.append(frame)
    history = N(prime=prime)
    def install(stack: Any, bootstrap: Any, replace: Any, **kwargs: Any) -> dict:
        return original(stack, bootstrap, replace, history_getter=lambda: history, **kwargs)
    monkeypatch.setattr(B, 'install', install)
    scenario(monkeypatch, tmp_path)
    assert calls == [100]
