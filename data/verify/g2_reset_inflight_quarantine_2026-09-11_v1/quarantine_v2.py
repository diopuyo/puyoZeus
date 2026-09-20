"""原journalのscope生成・epoch取得で、隔離対象の同call時計を再照合する。"""
from __future__ import annotations
from typing import Any
import quarantine as OLD

require = OLD.require


class Guard(OLD.Guard):
    def waiting(self, caller: Any) -> bool:
        r = self.recovery
        local = caller.f_locals
        if r.pending is not None and local.get('side') == '1P':
            item = r.journal.active
            require(item is not None and item['frame'] is caller, 'actual_J_caller')
            scope = r.journal.scope(r.pipe, '1P', local['frame_idx'], local['time_sec'])
            epoch = r.journal.epoch(r.pipe, '1P')
            require(item['scope'] == scope and item['epoch'] == epoch == r.pending['epoch'],
                    'item_scope_epoch_clock')
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
