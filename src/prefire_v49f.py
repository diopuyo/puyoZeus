"""全消し会計と決定的探索を使用する検証版。本番未接続・既定OFF。"""
from __future__ import annotations
from typing import Any
from src.prefire_v49e import PostV49E

PREFIRE_V49F_ENABLED_DEFAULT = False

class PostV49F(PostV49E):
    def __init__(self, predictor: Any, table: Any, policy: dict, native: Any,
                 enabled: bool = PREFIRE_V49F_ENABLED_DEFAULT, pre_model: dict | None = None) -> None:
        super().__init__(predictor, table, policy, native, enabled, pre_model,
                         aggregator='rep', all_variants=False, work_limit=100)
        self.defer_all_clear = enabled
