"""正常専用READYを元driver呼出codeへ束縛し、原会計/時刻/解放を維持する。"""
from types import FunctionType
from typing import Any
import normal_settings as K

READY = ('private_suffix_factory', 'hidden_probability_observer', 'provisional_context_observer')
FRAMES = tuple(range(K.FIRST, K.LAST + K.STRIDE, K.STRIDE))


def driver_class(original: Any) -> type:
    method = original.Driver.__call__
    selected = FunctionType(method.__code__, dict(method.__globals__, READY=READY),
                            method.__name__, method.__defaults__, method.__closure__)
    selected.__kwdefaults__ = method.__kwdefaults__
    class NormalDriver(original.Driver):
        original_call = selected
        def __call__(self, bridge: Any, frame: int) -> None:
            original.require(bridge.frames == FRAMES and self.frames == K.EARLIEST, 'normal_run_bounds')
            selected(self, bridge, frame)
    return NormalDriver
