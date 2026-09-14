"""1P到来専用型を2Pへ暗黙流用せず、対応済み2P物理型を明示的に束縛する。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def select(first: Any, supplied: Any, arrival: Any) -> Any:
    """新機構の1P所有を確認し、旧2Pの原Native/数学経路を保持する。"""
    require = arrival.B.require
    require(type(first) is arrival.Mode and supplied.Mode is arrival.Mode,
            'second_physical_first_type')
    require(first.connection.binding.scope[-1] == '1P', 'second_physical_first_side')
    require(arrival.BASE.B is arrival.B and issubclass(arrival.Mode, arrival.BASE.Mode),
            'second_physical_shared_belief')
    # 到来型は専用FIFO hook/保存先/起動資格が必要。2Pには未接続の型を渡さない。
    return SimpleNamespace(Mode=arrival.BASE.Mode)


def session_class(original: Any, arrival: Any) -> type:
    first_type, second_type = arrival.Mode, arrival.BASE.Mode
    class Session(original.Session):
        def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                     contract: Any, members: Any, frames: tuple[int, ...]) -> None:
            first = context['state']['probabilistic_tracking_mode']
            arrival.B.require(arrival.Mode is first_type and arrival.BASE.Mode is second_type,
                              'second_physical_type_changed')
            selected = select(first, physical, arrival)
            super().__init__(stack, context, policy, selected, contract, members, frames)
            self._completion_context, self._completion_first = context, first
            self.second_physical_selection = dict(
                first_type=type(first).__module__ + '.' + type(first).__qualname__,
                second_type=selected.Mode.__module__ + '.' + selected.Mode.__qualname__,
                first_arrival_source_shared=False, original_second_native_path=True,
                quality_gate_clear=False)

        def basis(self) -> None:
            first = self._completion_context['state']['probabilistic_tracking_mode']
            arrival.B.require(first is self._completion_first and arrival.Mode is first_type
                              and arrival.BASE.Mode is second_type, 'second_physical_type_changed')
            select(first, SimpleNamespace(Mode=first_type), arrival)
            return super().basis()
    return Session
