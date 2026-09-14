"""精算済み実rawの消去cellだけ、原mergeの旧履歴による復元から除く。"""
from __future__ import annotations
import copy
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import CodeType
from typing import Any

ROWS, COLS, EMPTY, UNKNOWN = 13, 6, 0, 10
SM_SHA = '5cde1678718bbde42987034fde4f0029055e3ce5f305c5745cb8731f620f9770'


@lru_cache(maxsize=1)
def expected(path: str) -> tuple[Any, Any]:
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SM_SHA, 'settled_SM_source'
    module = compile(data, path, 'exec', dont_inherit=True)
    merge = next(c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == '_merge_diff_only')
    cls = next(c for c in module.co_consts if isinstance(c, CodeType) and c.co_name == 'BoardStateMachine')
    apply = next(c for c in cls.co_consts if isinstance(c, CodeType) and c.co_name == '_apply_transition')
    return merge, apply


def key(board: Any) -> tuple[Any, ...]:
    return tuple(tuple(board.get(r, c) for c in range(COLS)) for r in range(ROWS))


def transition(sm: Any, signals: Any, target: Any, original: Any) -> dict[str, Any]:
    module = sys.modules[type(sm).__module__]
    assert signals.effect_gate_window_active is False and signals.is_match_active
    assert key(signals.cnn_board) == target and all(UNKNOWN not in row for row in target)
    assert sm.context.state.value != 'stable'
    baseline, merge, events = sm.context.confirmed_board, module._merge_diff_only, []
    merge_code, apply_code = expected(module.__file__)
    assert merge.__code__ == merge_code and original.__code__ == apply_code, 'settled_original_code'
    assert merge.__globals__ is original.__globals__ is vars(module), 'settled_original_globals'
    def qualified(old: Any, raw: Any, **kwargs: Any) -> Any:
        caller = sys._getframe(1)
        assert caller.f_code is original.__code__ and caller.f_locals['self'] is sm
        assert caller.f_locals['signals'] is signals and old is baseline and raw is signals.cnn_board
        assert not events and key(raw) == target
        history = kwargs.get('history_board')
        removed = []
        if history is not None:
            assert history is baseline
            filtered = history.copy()
            for r in range(ROWS):
                for c in range(COLS):
                    if target[r][c] == EMPTY and filtered.get(r, c) != EMPTY:
                        removed.append((r, c, filtered.get(r, c)))
                        filtered.set(r, c, EMPTY)
            kwargs['history_board'] = filtered
        events.append(dict(restoration_exclusions=removed))
        return merge(old, raw, **kwargs)
    module._merge_diff_only = qualified
    try:
        original(sm, module.BoardState.STABLE, signals)
    finally:
        module._merge_diff_only = merge
    assert len(events) == 1 and sm.context.state is module.BoardState.STABLE
    assert key(sm.context.confirmed_board) == target, 'settled_original_merge_mismatch'
    return events[0]


def apply(sm: Any, signals: Any, target: Any, original: Any) -> dict[str, Any]:
    """資格は呼出側で確認。ここは原mergeを私有preview後に一回適用する部品。"""
    shadow, observed = copy.deepcopy(sm), copy.deepcopy(signals)
    proposal = transition(shadow, observed, target, original)
    actual = transition(sm, signals, target, original)
    assert proposal == actual
    return dict(preview=proposal, actual=actual, physical_certified=False)
