"""元prefix installの戻り値だけを借用する。別Liveや偽frameは生成しない。"""
from pathlib import Path
from typing import Any

PREFIX = Path(__file__).resolve().parent.parent / 'g2_prefix_lane_integration_2026-09-13_v1'


class Lease:
    def __init__(self) -> None:
        self.live: Any = None
        self.adapter: Any = None
        self.closed = False
        self.closed_modes: dict[int, Any] = {}

    def close(self) -> None:
        self.closed = True
        self.live = self.adapter = None
        self.closed_modes.clear()

    def current(self) -> tuple[Any, Any]:
        if self.closed:
            raise ValueError('prefix_lease_closed')
        if self.live is None:
            raise ValueError('prefix_lease_not_installed')
        if type(self.live) is not self.adapter.Live:
            raise ValueError('prefix_lease_type_changed')
        return self.live, self.adapter

    def mode_open(self, mode: Any) -> None:
        live, _ = self.current()
        if not isinstance(mode, live.module.Mode):
            raise ValueError('prefix_lease_mode_type')
        if id(mode) in self.closed_modes or mode.stream.closed or mode.error is not None:
            raise ValueError('prefix_lease_mode_closed_or_failed')

    def capture_close(self, stack: Any, cls: type, patch: Any) -> None:
        original = cls.close
        def close(mode: Any) -> None:
            try:
                original(mode)
            finally:
                self.closed_modes[id(mode)] = mode
        patch(stack, cls, 'close', close)


def install(stack: Any, selection: Any, replace: Any) -> Lease:
    """A23.UNBOUND_MODULEへcold開始前に装着し、元の一回の生成を捕捉する。"""
    original = selection.dependencies
    if Path(original.__code__.co_filename).resolve() != PREFIX / 'prefix_selection.py':
        raise ValueError('prefix_lease_selection_source')
    lease = Lease()
    stack.callback(lease.close)
    installed: list[Any] = []

    def dependencies(load: Any) -> Any:
        adapter = original(load)
        if installed:
            if adapter is not installed[0]:
                raise ValueError('prefix_lease_adapter_changed')
            return adapter
        prior = adapter.install
        if Path(prior.__code__.co_filename).resolve() != PREFIX / 'prefix_live_adapter.py':
            raise ValueError('prefix_lease_install_source')

        def capture(inner: Any, module: Any, patch: Any) -> Any:
            if lease.closed or lease.live is not None:
                raise ValueError('prefix_lease_repeated_install')
            live = prior(inner, module, patch)
            lease.live, lease.adapter = live, adapter
            inner.callback(lease.close)  # 元Liveの解放より先に借用を無効化する。
            lease.capture_close(inner, module.Mode, patch)
            lease.current()
            return live

        replace(stack, adapter, 'install', capture)
        installed.append(adapter)
        return adapter

    replace(stack, selection, 'dependencies', dependencies)
    return lease
