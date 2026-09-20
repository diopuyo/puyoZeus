"""2P原基準・原Jを既存の確率連鎖/通常手反映へ接続する。"""
from __future__ import annotations

import io
from types import SimpleNamespace as N
from typing import Any
import second_tracking as T
import journal_context as C


class Mode(T.Mode):
    def __init__(self, evidence: Any, witness: Any, pipe: Any, registry: Any,
                 factory: Any, deadline: int, policy: Any, physical: Any, provider: Any) -> None:
        C.require(provider.journal is witness.journal, 'second_physical_provider_owner')
        super().__init__(evidence,witness,pipe,registry,factory,deadline,policy)
        c = self.connection
        c.recovery.provider, c.recovery.state = provider, evidence.state
        # 初回観測のframeのみ参照。reset/落下gateが発行したとは偽装しない。
        c.observer = N(gate=N(candidate=registry.current(c.binding)))
        self.physical = physical.Mode(c,evidence.state,io.StringIO())
        self.physical.native = self.native
        self.applied = self.physical.applied
        self.latest: dict[str,Any] | None = None

    def observe(self, item: Any, error: Any, result: Any = None) -> Any:
        C.require(not self.closed and self.error is None and self.evidence.error is None
                  and not self.evidence.closed, 'second_physical_lifetime')
        # 既存Mode.observeが原Jを一度だけ読む。親observeとの二重採録をしない。
        self.latest = self.physical.observe(item,result,error)
        return self.latest
