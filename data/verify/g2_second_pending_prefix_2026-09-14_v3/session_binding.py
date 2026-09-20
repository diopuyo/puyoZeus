"""既存Sessionの所有検査後に2P型だけ選び、外側終了でsidecarを閉じる。"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import physical_adapter as A
import mode_composition as M


def finish_modes(session: Any, expected: type | None) -> BaseException | None:
    failure = None
    owned = list(getattr(session,'_prefix_owned',()))
    for mode in getattr(session,'modes',()):
        if not any(mode is prior for prior,_ in owned):
            owned.append((mode,mode.physical))
    for mode,physical in owned:
        try:
            if expected is None or type(physical) is not expected:
                raise ValueError('second_prefix_session_foreign_physical')
            if mode.physical is not physical and failure is None:
                failure = ValueError('second_prefix_session_physical_replaced')
            physical.prefix.finish()
        except BaseException as error:
            if failure is None:
                failure = error
    return failure


def wrap_factory(previous: Any) -> Any:
    def factory(original: Any, arrival: Any) -> type:
        base = previous(original,arrival)
        engine, serializer = arrival.C.T, arrival.BASE.V1.S
        class Session(base):
            def __init__(self, stack: Any, context: Any, policy: Any, physical: Any,
                         contract: Any, members: Any, frames: tuple) -> None:
                self._prefix_mode_type = None
                self._prefix_owned: list[tuple] = []
                super().__init__(stack,context,policy,physical,contract,members,frames)
                arrival.B.require(not self.modes and self.mode is None, 'second_prefix_session_order')
                selected = M.compose(self.physical.Mode,arrival,A,engine,serializer)
                self._prefix_mode_type = selected
                self.physical = SimpleNamespace(Mode=selected)
                stack.push(self.finish_prefix)
                self.second_physical_selection.update(second_prefix_candidate=True,
                    ordinary_second_path_preserved=True, original_second_native_path=True,
                    quality_gate_clear=False)

            def basis(self) -> None:
                super().basis()
                for mode in self.modes:
                    if not any(mode is prior for prior,_ in self._prefix_owned):
                        arrival.B.require(type(mode.physical) is self._prefix_mode_type,
                                          'second_prefix_basis_physical_type')
                        self._prefix_owned.append((mode,mode.physical))

            def finish_prefix(self, kind: Any, body: Any, trace: Any) -> bool:
                failure = finish_modes(self,self._prefix_mode_type)
                if failure is not None and getattr(self,'error',None) is None:
                    self.error = failure
                # 最初の例外を保持し、元Sessionを含む後続の解放はExitStackへ委ねる。
                if body is None and failure is not None:
                    raise failure
                return False
        return Session
    return factory
