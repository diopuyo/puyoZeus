"""保持証拠の再認可でも現在の原J enqueueとaddedを照合する最小差分。"""
from __future__ import annotations
from types import FunctionType
from typing import Any
import held as OLD

Held, require = OLD.Held, OLD.require
ORIGINAL_CHECK = OLD.check


def check(control: Any, held: Held, binding: Any, item: Any, view: Any) -> None:
    ORIGINAL_CHECK(control, held, binding, item, view)
    provider, side = control.provider, view.scope[-1]
    row = provider.link.current(view)
    journal = provider.journal
    scope = journal.scope(row['pipe'], side, view.frame, view.clock)
    owner = journal.fifo.entries[(id(row['pipe']), side)]
    enqueue = provider.enqueues.get(side)
    provider._parts.V.Provider.check_enqueue(provider, enqueue, scope, row['epoch'], owner)
    require(tuple(enqueue['added_occurrence_tokens']) == view.added, 'current_J_added')


# 旧候補の型・一回性・source時計を保持し、再認可の検査だけを強化する。
capture = FunctionType(OLD.capture.__code__, dict(vars(OLD), check=check))
authorize = FunctionType(OLD.authorize.__code__, dict(vars(OLD), check=check))
ready, from_call, consume = OLD.ready, OLD.from_call, OLD.consume
