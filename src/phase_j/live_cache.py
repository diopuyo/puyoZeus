"""ライブ限定の連鎖キャッシュ。古い盤面を退避して新しい試合の再用を保つ。"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import gc
from typing import Any, Iterator
from unittest.mock import patch

from src.chain import ChainSimulator

LIVE_CACHE_SIZE = 2048


class RecentCache(OrderedDict):
    def __init__(self, capacity: int = LIVE_CACHE_SIZE) -> None:
        super().__init__()
        self.capacity = capacity
        self.hits = self.misses = self.evictions = 0

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self.misses += 1
            return default
        self.hits += 1
        self.move_to_end(key)
        return self[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)
            self.evictions += 1


@contextmanager
def bounded_chain_caches() -> Iterator[None]:
    """既存singletonと今後の生成を対象にし、offlineのクラス既定値を変えない。"""
    original = ChainSimulator.__init__
    existing = [o for o in gc.get_objects() if type(o) is ChainSimulator]
    previous = [(o, o._cache) for o in existing]
    for simulator in existing:
        simulator._cache = RecentCache()

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self._cache = RecentCache()

    try:
        with patch.object(ChainSimulator, '__init__', initialize):
            yield
    finally:
        for simulator, cache in previous:
            simulator._cache = cache
