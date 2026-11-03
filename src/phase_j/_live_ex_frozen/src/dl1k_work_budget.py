"""EX専用の決定的な探索量。旗も上限も明示し、本番の既定値は変えない。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from src.phase_j._live_ex_frozen.src.prefire_search_v410 import SearchBudgetExpired

FLAG = 'PUYO_EX_WORK_BUDGET'
LIMIT_ENV = 'PUYO_EX_WORK_LIMIT'
WIDTH = 1
SAMPLES = 2
COUNTED_CALLS = ('enumerate_and_simulate_placements_py', 'ama_search_py')


def configured_limit(enabled: bool = False, limit: int | None = None) -> int | None:
    """分布を測る前に仮の上限を製品値にしない。ON時は正の整数を必須とする。"""
    if not enabled and os.environ.get(FLAG, '0') != '1':
        return None
    if limit is None:
        value = os.environ.get(LIMIT_ENV, '')
        if not value.isdecimal():
            raise ValueError(f'{FLAG}=1 には {LIMIT_ENV} の正の整数が必要')
        limit = int(value)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError('仕事量の上限は正の整数')
    return limit


@dataclass
class WorkBudget:
    """1単位=配置列挙または幅1・最大深さ3のama呼出し1回。全テープ共通。"""
    limit: int | None = None
    used: int = 0
    exhausted: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    width: int = WIDTH
    samples: int = SAMPLES

    def check(self) -> None:
        if self.exhausted:
            raise SearchBudgetExpired('EX探索の仕事量上限')

    def charge(self, name: str) -> None:
        self.check()
        if self.limit is not None and self.used >= self.limit:
            self.exhausted = True
            self.check()
        self.used += 1
        self.counts[name] = self.counts.get(name, 0) + 1


class CountedNative:
    """固定幅・深さのnative入口で数える。既存APIと結果の順序は変えない。"""

    def __init__(self, native: Any, work: WorkBudget) -> None:
        self.native, self.work = native, work

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.native, name)
        if name not in COUNTED_CALLS:
            return value

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.work.charge(name)
            return value(*args, **kwargs)

        return counted
