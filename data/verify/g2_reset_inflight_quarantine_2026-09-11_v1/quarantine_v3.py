"""不正callerのlocal欠測も原呼出の前に理由付きで拒否する。"""
from __future__ import annotations
from typing import Any
import quarantine_v2 as V2

require = V2.require


class Guard(V2.Guard):
    def waiting(self, caller: Any) -> bool:
        if self.recovery.pending is not None and caller.f_locals.get('side') == '1P':
            require(caller.f_code in self.recovery.journal.codes, 'actual_J_caller')
            require(all(key in caller.f_locals for key in ('frame_idx', 'time_sec')), 'caller_clock_missing')
        return super().waiting(caller)


def install(stack: Any, recovery: Any, namespace: dict[str, Any]) -> Guard:
    original = namespace['infer_placement']
    value = Guard(recovery, original)
    wrapper = value.infer
    namespace['infer_placement'] = wrapper
    def restore() -> None:
        require(namespace['infer_placement'] is wrapper, 'restore_binding_changed')
        namespace['infer_placement'] = original
    stack.callback(restore)
    return value
