"""基準取得までは原selectedの直接callerを保持する接続版。"""
from __future__ import annotations
from types import MethodType
from typing import Any
import tracking_mode as V1

B, Mode, patch, KEY = V1.B, V1.Mode, V1.patch, V1.KEY


def install(stack: Any, connection: Any, state: dict[str, Any]) -> Mode:
    B.require(KEY not in state, 'duplicate_tracking_mode')
    stream = stack.enter_context((state['output'] / 'PROBABILISTIC_TRACKING.jsonl').open('x', encoding='utf-8'))
    value = Mode(connection, state, stream)
    state[KEY] = value
    stack.callback(value.close)
    r, sink = connection.recovery, state['live_history_sink']
    complete, selected, before, after = V1.handlers(value, r, sink)
    installed = False
    def completed(self: Any, item: Any, result: Any, error: Any) -> Any:
        nonlocal installed
        returned = complete(item, result, error)
        if value.native is not None and not installed:
            # 原Recovery.selectedはController.updateからの直接呼出を検査する。
            # 待機終了後だけ切替え、そのチェックの深さや許可callerを緩めない。
            patch(stack, r.provider, 'selected', selected)
            installed = True
        return returned
    patch(stack, r, 'complete', MethodType(completed, r))
    patch(stack, sink, 'before', before)
    patch(stack, sink, 'after', after)
    return value
