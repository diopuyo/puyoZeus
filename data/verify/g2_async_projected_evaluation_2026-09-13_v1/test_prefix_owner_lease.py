"""元prefix cold install一回の捕捉・未到達拒否・退出解放。動画検収とは別。"""
from contextlib import ExitStack
from typing import Any
import sys
import pytest
import prefix_owned_adapter as A
import prefix_owner_lease as L


def test_original_install_capture_and_exit() -> None:
    original = A.UNBOUND_MODULE.dependencies
    with ExitStack() as stack:
        lease = L.install(stack, A.UNBOUND_MODULE, A.A.A.A.V4.replace_owned)
        with pytest.raises(ValueError, match='not_installed'):
            lease.current()
        A.configured(stack)
        import probe_native_merge
        owner = sys.modules[A.A.A.A.V4.OWNED_ALIAS]
        parts = owner.dependencies().modules()
        live, adapter = lease.current()
        assert type(live) is adapter.Live and live.module is parts.mode
        assert not live.active and not live.lanes
        assert owner.dependencies().modules().mode is parts.mode
        assert lease.current()[0] is live
        with ExitStack() as inner, pytest.raises(ValueError, match='repeated_install'):
            adapter.install(inner, parts.mode, A.A.A.A.V4.replace_owned)
        assert lease.current()[0] is live
    assert lease.closed and lease.live is None and lease.adapter is None
    assert not live.lanes and not live.active and not live.recorder.owners
    assert A.UNBOUND_MODULE.dependencies is original
    with pytest.raises(ValueError, match='closed'):
        lease.current()


def test_unreached_exit_and_foreign_selection() -> None:
    original = A.UNBOUND_MODULE.dependencies
    failure = RuntimeError('planned_owner_body')
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as stack:
            lease = L.install(stack, A.UNBOUND_MODULE, A.A.A.A.V4.replace_owned)
            raise failure
    assert caught.value is failure and lease.closed
    assert A.UNBOUND_MODULE.dependencies is original
    from types import SimpleNamespace
    with ExitStack() as stack, pytest.raises(ValueError, match='selection_source'):
        L.install(stack, SimpleNamespace(dependencies=lambda load: None), None)


@pytest.mark.parametrize('body_failure', [False, True])
def test_mode_close_invalidates_before_stream_close(body_failure: bool) -> None:
    from types import SimpleNamespace as N
    failure = RuntimeError('original_close_failure')
    class Mode:
        def close(self) -> None:
            if body_failure:
                raise failure
    class Live:
        module = N(Mode=Mode)
    lease = L.Lease()
    lease.live, lease.adapter = Live(), N(Live=Live)
    mode = Mode()
    mode.stream, mode.error = N(closed=False), None
    def patch(stack: Any, obj: Any, name: str, value: Any) -> None:
        stack.callback(setattr, obj, name, getattr(obj, name))
        setattr(obj, name, value)
    with ExitStack() as stack:
        lease.capture_close(stack, Mode, patch)
        lease.mode_open(mode)
        if body_failure:
            with pytest.raises(RuntimeError) as caught:
                mode.close()
            assert caught.value is failure
        else:
            mode.close()
        assert not mode.stream.closed
        with pytest.raises(ValueError, match='mode_closed_or_failed'):
            lease.mode_open(mode)
    lease.close()
    assert not lease.closed_modes
